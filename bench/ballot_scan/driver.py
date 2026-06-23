#!/usr/bin/env python
"""Correctness + speedup sweep for the ballot/popcount boolean add-scan.

Path selection is purely STRUCTURAL (no runtime knob): a provably-{0,1}
integer-add scan lowers to ballot+popcount; anything else stays on the generic
shuffle scan. We therefore drive the two paths by the operand, not an env var:

  worker --mask 1 : scan x&1   -> ballot fast path (fires ctpop)
  worker --mask 0 : scan ints  -> generic shuffle scan (must NOT fire ctpop)

Correctness phase (rep=1): assert each path matches torch.cumsum, and assert the
structural invariant that the generic path NEVER fires the fast path. (A true
same-input A/B of the two *lowerings* of one kernel would require building an
unpatched checkout; here we validate each path against torch ground truth, which
is stronger than a path-vs-path hash compare.)

Speed phase (rep large, big grid): the mask=1 and mask=0 kernels are identical
except for the warp-scan algorithm (ballot vs shuffle) plus a trivial per-element
`& 1` charged to the ballot side, so speedup = generic_ms / ballot_ms is a
conservative in-build estimate of the fast path's win.
"""
import os, sys, json, subprocess, concurrent.futures as cf

PY = os.environ.get("BALLOT_PY", sys.executable)
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
OUT = os.environ.get("BALLOT_OUT", "/tmp/ballot")


def run_worker(cfg, time=False, cache_root=None):
    cache_root = cache_root or os.path.join(OUT, "cache")
    env = dict(os.environ)
    env["TRITON_ALWAYS_COMPILE"] = "1"
    tag = (f"{cfg['M']}x{cfg['N']}_a{cfg['axis']}_w{cfg['num_warps']}_{cfg['dtype']}"
           f"_r{cfg['reverse']}_m{cfg['mask']}")
    env["TRITON_CACHE_DIR"] = os.path.join(cache_root, tag)
    cmd = [PY, WORKER,
           "--M", str(cfg["M"]), "--N", str(cfg["N"]), "--axis", str(cfg["axis"]),
           "--num-warps", str(cfg["num_warps"]), "--dtype", cfg["dtype"],
           "--reverse", str(cfg["reverse"]), "--mask", str(cfg["mask"]),
           "--rep", str(cfg.get("rep", 1)), "--time", "1" if time else "0",
           "--grid", str(cfg.get("grid", 256))]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": "timeout", "fired": None}
    line = ""
    for l in out.stdout.strip().splitlines()[::-1]:
        if l.startswith("{"):
            line = l
            break
    if not line:
        return {"ok": False, "err": "no-json: " + out.stderr.strip()[-300:], "fired": None}
    return json.loads(line)


def correctness():
    os.makedirs(OUT, exist_ok=True)
    dtypes = ["int8", "int16", "int32", "int64"]
    shapes_axes = [
        # (M, N, axis): probe broadcast, full-warp, multi-warp, non-axis, multi-block
        (1, 2, 1), (1, 4, 1), (1, 8, 1), (1, 16, 1), (1, 32, 1),
        (1, 64, 1), (1, 128, 1), (1, 256, 1), (1, 1024, 1),
        (2, 1, 0), (8, 1, 0), (32, 1, 0), (64, 1, 0), (128, 1, 0),
        (8, 32, 1), (16, 32, 1), (32, 16, 1), (32, 32, 1), (32, 32, 0),
        (8, 8, 1), (8, 8, 0), (64, 64, 1), (64, 64, 0),
        (2, 1024, 1), (1024, 2, 0), (4, 256, 1), (256, 4, 0),
    ]
    warps = [1, 2, 4, 8]
    configs = []
    # mask=1 (ballot candidates) across dtypes, both reverse, representative warps
    for (M, N, axis) in shapes_axes:
        for dt in dtypes:
            for rev in (0, 1):
                for w in warps:
                    configs.append(dict(M=M, N=N, axis=axis, num_warps=w,
                                        dtype=dt, reverse=rev, mask=1, rep=1, grid=64))
    # mask=0 (generic path) correctness + the "never fires" invariant
    for (M, N, axis) in [(1, 256, 1), (32, 32, 1), (64, 64, 0), (8, 32, 1), (1, 32, 1)]:
        for w in (1, 4):
            configs.append(dict(M=M, N=N, axis=axis, num_warps=w, dtype="int32",
                                reverse=0, mask=0, rep=1, grid=64))
    # dedup
    seen = set(); uniq = []
    for c in configs:
        k = tuple(sorted(c.items()))
        if k not in seen:
            seen.add(k); uniq.append(c)
    print(f"# correctness configs: {len(uniq)}")

    results = []
    npass = nfail = nfired = nfallback = ninvariant_fail = 0
    fails = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for c, r in zip(uniq, ex.map(lambda c: run_worker(c), uniq)):
            ok = r.get("ok")
            refok = ok and (r.get("ref_ok") in (True, None))
            fired = bool(r.get("fired"))
            # invariant: the generic path (mask=0) must NEVER fire the fast path
            invariant_ok = not (c["mask"] == 0 and fired)
            if not invariant_ok:
                ninvariant_fail += 1
            good = ok and refok and invariant_ok
            if c["mask"] == 1:
                nfired += fired; nfallback += (not fired)
            if good:
                npass += 1
            else:
                nfail += 1; fails.append((c, r))
            results.append(dict(cfg=c, res=r, refok=refok, fired=fired, good=good))
    print(f"PASS={npass} FAIL={nfail}  | mask=1 fired={nfired} fellback={nfallback} "
          f"| generic-never-fires violations={ninvariant_fail}")
    if fails:
        print("\n=== FAILURES ===")
        for c, r in fails[:40]:
            print(f"  {c}")
            print(f"    ok={r.get('ok')} fired={r.get('fired')} ref={r.get('ref_ok')}/"
                  f"{r.get('ref_maxdiff')} err={r.get('err','')}")
    json.dump(results, open(os.path.join(OUT, "correctness.json"), "w"), indent=1)
    print("\n=== fast-path coverage (mask=1, dtype=int32, w=4, rev=0) ===")
    for r in results:
        c = r["cfg"]
        if c["mask"] == 1 and c["dtype"] == "int32" and c["num_warps"] == 4 and c["reverse"] == 0:
            print(f"  {c['M']:>4}x{c['N']:<4} axis={c['axis']}  fired={r['fired']}  good={r['good']}")
    return nfail == 0


def speed():
    # rep=1  -> realistic end-to-end (load/store/launch bound, Amdahl-diluted)
    # rep=64 -> scan-isolated (amplifies the warp-scan cost) across configs.
    # speedup = generic_ms / ballot_ms: identical kernels but for the warp-scan
    # algorithm (+ a trivial `& 1` charged to the ballot side -> conservative).
    os.makedirs(OUT, exist_ok=True)
    grid = 4096
    base = [
        (1, 32, 1, "full-warp 32"),
        (1, 16, 1, "broadcast 16"),
        (1, 8, 1, "broadcast 8"),
        (1, 4, 1, "broadcast 4"),
        (1, 64, 1, "2 warps-worth (64)"),
        (1, 128, 1, "axis 128"),
        (1, 256, 1, "axis 256"),
        (1, 1024, 1, "axis 1024"),
        (32, 32, 1, "2D 32x32 ax1"),
        (32, 32, 0, "2D 32x32 ax0"),
        (8, 32, 1, "2D 8x32 ax1"),
    ]
    warps = [1, 4, 8]
    reps = [1, 64]
    print(f"\n# speed sweep grid={grid}  reps={reps}  (speedup = generic_ms / ballot_ms)")
    print(f"{'shape/axis':<22}{'rep':<5}{'warps':<6}{'fired':<6}{'ballot ms':<11}"
          f"{'generic ms':<12}{'speedup':<9}{'ok'}")
    rows = []
    for (M, N, axis, label) in base:
        for rep in reps:
            for w in warps:
                cb = dict(M=M, N=N, axis=axis, num_warps=w, dtype="int32", reverse=0,
                          mask=1, rep=rep, grid=grid)
                cg = dict(cb, mask=0)
                b = run_worker(cb, time=True)
                g = run_worker(cg, time=True)
                ok = b.get("ok") and g.get("ok")
                sp = (g["ms"] / b["ms"]) if (b.get("ms") and g.get("ms")) else None
                fired = b.get("fired")
                rows.append(dict(label=label, rep=rep, w=w, fired=fired,
                                 bms=b.get("ms"), gms=g.get("ms"), sp=sp, ok=ok))
                spn = f"{sp:.2f}x" if sp else "-"
                bms = f"{b['ms']:.4f}" if b.get("ms") else "-"
                gms = f"{g['ms']:.4f}" if g.get("ms") else "-"
                print(f"{label:<22}{rep:<5}{w:<6}{str(fired):<6}{bms:<11}{gms:<12}{spn:<9}{ok}")
    json.dump(rows, open(os.path.join(OUT, "speed.json"), "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    ok = True
    if mode in ("correctness", "both"):
        ok = correctness()
    if mode in ("speed", "both"):
        speed()
    sys.exit(0 if ok else 1)
