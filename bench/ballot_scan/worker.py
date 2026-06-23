#!/usr/bin/env python
"""Compile + run + (optionally) time ONE scan config in ONE code path.

Path selection is purely STRUCTURAL: a scan of a provably-{0,1} integer-add
operand lowers to the ballot+popcount fast path; anything else lowers to the
generic shuffle scan. There is no runtime toggle. We therefore exercise each
path by its INPUT, not an env var:

  --mask 1 : scan `x & 1` (provably {0,1})      -> ballot+popcount fast path
  --mask 0 : scan raw multi-valued ints         -> generic shuffle scan

We dump LLVM IR and report whether `llvm.intr.ctpop` is present (`fired`) so the
driver can assert the fast path fires for boolean operands and NEVER for generic
ones. Correctness is checked against torch.cumsum for both paths (rep==1).

Outputs a single JSON line on stdout.
"""
import os, sys, json, argparse, hashlib
import torch
import triton
import triton.language as tl


@triton.jit
def scan_kernel(X, Z, BM: tl.constexpr, BN: tl.constexpr, AXIS: tl.constexpr,
                REV: tl.constexpr, DT: tl.constexpr, REP: tl.constexpr,
                MASK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    off = rm[:, None] * BN + rn[None, :]
    x = tl.load(X + off).to(DT)
    # MASK=1: force a provably-{0,1} operand -> ballot fast path.
    # MASK=0: scan the raw ints -> generic shuffle path (gate refuses it).
    z = (x & 1) if MASK else x
    for _ in tl.static_range(REP):
        # Re-mask each iteration (MASK=1) so the scan stays boolean and the fast
        # path keeps firing; REP only amplifies scan cost for timing. The extra
        # `& 1` is charged to the ballot path, so any measured speedup is if
        # anything conservative.
        z = tl.cumsum(z & 1, axis=AXIS, reverse=REV) if MASK \
            else tl.cumsum(z, axis=AXIS, reverse=REV)
    tl.store(Z + off, z)


DT_MAP = {
    "int8": (tl.int8, torch.int8),
    "int16": (tl.int16, torch.int16),
    "int32": (tl.int32, torch.int32),
    "int64": (tl.int64, torch.int64),
}


def torch_ref(x_int, axis, reverse, out_dtype, mask):
    # One cumsum (REP==1 semantics) over the same operand the kernel scans.
    b = x_int.to(torch.int64)
    if mask:
        b = b & 1
    if reverse:
        b = torch.flip(b, [axis])
    r = torch.cumsum(b, dim=axis)
    if reverse:
        r = torch.flip(r, [axis])
    # mimic wrap-around of the narrow result type
    bits = {torch.int8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64}[out_dtype]
    if bits < 64:
        r = r & ((1 << bits) - 1)
    return r.to(out_dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--axis", type=int, required=True)
    ap.add_argument("--num-warps", type=int, required=True)
    ap.add_argument("--dtype", choices=list(DT_MAP), default="int32")
    ap.add_argument("--reverse", type=int, default=0)
    ap.add_argument("--mask", type=int, default=1,
                    help="1: scan x&1 (ballot fast path); 0: scan raw ints (generic path)")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--grid", type=int, default=2048)
    ap.add_argument("--out-file", default="")
    args = ap.parse_args()

    dev = "cuda"
    M, N, axis = args.M, args.N, args.axis
    tl_dt, th_dt = DT_MAP[args.dtype]
    torch.manual_seed(0)

    if args.mask:
        X = (torch.rand((M, N), device=dev) > 0.5).to(torch.int32)  # 0/1 ints
    else:
        # multi-valued -> NOT provably {0,1}, so the gate refuses the fast path
        X = torch.randint(2, 97, (M, N), device=dev, dtype=torch.int32)

    Z = torch.empty((M, N), device=dev, dtype=th_dt)

    fired = None
    try:
        compiled = scan_kernel.warmup(X, Z, BM=M, BN=N, AXIS=axis, REV=args.reverse,
                                      DT=tl_dt, REP=args.rep, MASK=args.mask,
                                      grid=(1,), num_warps=args.num_warps)
        llir = compiled.asm.get("llir", "")
        fired = ("@llvm.ctpop" in llir) or ("llvm.intr.ctpop" in llir)
    except Exception as e:
        print(json.dumps({**vars(args), "ok": False, "fired": None,
                          "err": f"compile: {type(e).__name__}: {e}"}))
        return

    grid = (args.grid,)
    try:
        scan_kernel[grid](X, Z, BM=M, BN=N, AXIS=axis, REV=args.reverse,
                          DT=tl_dt, REP=args.rep, MASK=args.mask, num_warps=args.num_warps)
        torch.cuda.synchronize()
    except Exception as e:
        print(json.dumps({**vars(args), "ok": False, "fired": fired,
                          "err": f"run: {type(e).__name__}: {e}"}))
        return

    # correctness vs torch is meaningful for a single cumsum (rep==1)
    ref_ok, ref_maxdiff = None, None
    if args.rep == 1:
        ref = torch_ref(X, axis, bool(args.reverse), th_dt, bool(args.mask))
        diff = (Z.to(torch.int64) - ref.to(torch.int64)).abs().max().item()
        ref_maxdiff = diff
        ref_ok = (diff == 0)

    h = hashlib.sha1(Z.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    if args.out_file:
        torch.save(Z.detach().cpu(), args.out_file)

    ms = None
    if args.time:
        fn = lambda: scan_kernel[grid](X, Z, BM=M, BN=N, AXIS=axis, REV=args.reverse,
                                       DT=tl_dt, REP=args.rep, MASK=args.mask,
                                       num_warps=args.num_warps)
        ms = triton.testing.do_bench(fn, warmup=50, rep=200)

    print(json.dumps({**vars(args), "ok": True, "fired": fired, "hash": h,
                      "ref_ok": ref_ok, "ref_maxdiff": ref_maxdiff, "ms": ms}))


if __name__ == "__main__":
    main()
