# Boolean add-scan ballot+popcount differential & benchmark harness

Verifies and benchmarks the `tt.scan` boolean add-scan -> ballot+popcount fast
path in `lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp`.

The fast path is toggled by `TRITON_DISABLE_BALLOT_SCAN` (raw getenv, not in the
Triton cache key), so each (config, mode) runs in a fresh subprocess with
`TRITON_ALWAYS_COMPILE=1`.

- `worker.py` — compile + run + (optionally) time ONE scan config in ONE mode.
  Emits a JSON line; reports whether the fast path fired (ctpop in LLVM IR),
  correctness vs torch, and timing.
- `driver.py correctness` — for every config run ballot vs forced-shuffle and
  assert bit-exact equality + torch reference; record fast-path coverage.
- `driver.py speed` — speedup = shuffle_ms / ballot_ms, rep in {1 (end-to-end),
  64 (scan-isolated)}.

Run (with the patched build active):
    python bench/ballot_scan/driver.py correctness
    python bench/ballot_scan/driver.py speed

Lit test: `test/Conversion/scan_ballot_to_llvm.mlir`.
