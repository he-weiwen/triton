// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm 2>&1 | FileCheck %s

// A boolean (0/1) integer add-scan lowers the warp scan to ballot+popcount
// (llvm.intr.ctpop); anything not provably {0,1} keeps the generic shuffle scan
// (no ctpop). Path selection is purely structural -- a function of the operand's
// producer chain and the combine op -- so the cases are distinguished by their
// input IR alone, with no runtime toggle. The {0,1} proof accepts only the
// producers measured to actually fire on real (repo + inductor) kernels: i1, 0/1
// constants, extui/trunci, `and` (the `x & 1` mask), and `select` (the
// `tl.where(c,1,0)` idiom). Data-movement ops (convert_layout/splat/broadcast/
// reshape/...) are intentionally NOT looked through, so a scan fed through one
// keeps the generic shuffle scan.

#l32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// extui(i1)->i32 proves the operand is {0,1} (the shape `tl.cumsum(bool)` emits).
// CHECK-LABEL: @bool_extui
// CHECK: llvm.intr.ctpop
tt.func private @bool_extui(%arg0: tensor<32xi1, #l32>) -> tensor<32xi32, #l32> {
  %b = arith.extui %arg0 : tensor<32xi1, #l32> to tensor<32xi32, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  tt.return %0 : tensor<32xi32, #l32>
}

// `x & 1` masks the operand to {0,1} regardless of x (AND only clears bits).
// CHECK-LABEL: @bool_and_mask
// CHECK: llvm.intr.ctpop
tt.func private @bool_and_mask(%arg0: tensor<32xi32, #l32>) -> tensor<32xi32, #l32> {
  %c1 = arith.constant dense<1> : tensor<32xi32, #l32>
  %b = arith.andi %arg0, %c1 : tensor<32xi32, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  tt.return %0 : tensor<32xi32, #l32>
}

// `select` between {0,1} arms is {0,1} (the `tl.where(c, 1, 0)` idiom).
// CHECK-LABEL: @bool_select
// CHECK: llvm.intr.ctpop
tt.func private @bool_select(%cond: tensor<32xi1, #l32>) -> tensor<32xi32, #l32> {
  %c0 = arith.constant dense<0> : tensor<32xi32, #l32>
  %c1 = arith.constant dense<1> : tensor<32xi32, #l32>
  %b = arith.select %cond, %c1, %c0 : tensor<32xi1, #l32>, tensor<32xi32, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  tt.return %0 : tensor<32xi32, #l32>
}

// A reverse boolean add-scan still fires: the reverse direction is handled by
// the flip(scan(flip)) wrapper in emitFastScan, so warpScanBallot runs a forward
// scan over the flipped data and the ballot fast path applies (the old
// !getReverse() gate was over-conservative).
// CHECK-LABEL: @bool_reverse
// CHECK: llvm.intr.ctpop
tt.func private @bool_reverse(%arg0: tensor<32xi1, #l32>) -> tensor<32xi32, #l32> {
  %b = arith.extui %arg0 : tensor<32xi1, #l32> to tensor<32xi32, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  tt.return %0 : tensor<32xi32, #l32>
}

// Element width is not gated: the popcount is computed in i32 and zext'd up to
// the (wider) element type. i64 is the width TorchInductor's add-scans use.
// CHECK-LABEL: @bool_extui_i64
// CHECK: llvm.intr.ctpop
tt.func private @bool_extui_i64(%arg0: tensor<32xi1, #l32>) -> tensor<32xi64, #l32> {
  %b = arith.extui %arg0 : tensor<32xi1, #l32> to tensor<32xi64, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i64, %c: i64):
    %1 = arith.addi %a, %c : i64
    tt.scan.return %1 : i64
  }) : (tensor<32xi64, #l32>) -> tensor<32xi64, #l32>
  tt.return %0 : tensor<32xi64, #l32>
}

// Narrow width: the i32 popcount is trunc'd to the element type. For i1, trunc
// yields count mod 2 -- which is exactly the i1 add-scan's wraparound, so the
// fast path is still correct.
// CHECK-LABEL: @bool_i1
// CHECK: llvm.intr.ctpop
tt.func private @bool_i1(%arg0: tensor<32xi1, #l32>) -> tensor<32xi1, #l32> {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i1, %c: i1):
    %1 = arith.addi %a, %c : i1
    tt.scan.return %1 : i1
  }) : (tensor<32xi1, #l32>) -> tensor<32xi1, #l32>
  tt.return %0 : tensor<32xi1, #l32>
}

// extsi is intentionally NOT a supported producer: it has zero measured
// real-world incidence, and extsi(i1) is the {0,-1} miscompile trap (an i1
// "true" sign-extends to all-ones, not 1). So even an extsi of an otherwise
// proven-{0,1} value must NOT fire -- it falls back to the generic scan.
// CHECK-LABEL: @not_bool_extsi
// CHECK-NOT: llvm.intr.ctpop
tt.func private @not_bool_extsi(%arg0: tensor<32xi1, #l32>) -> tensor<32xi32, #l32> {
  %b = arith.extsi %arg0 : tensor<32xi1, #l32> to tensor<32xi32, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  tt.return %0 : tensor<32xi32, #l32>
}

// Generic (non-boolean) add-scan: operand is not provably {0,1}, so the fast
// path must NOT fire (this is the last scan func; CHECK-NOT runs to EOF).
// CHECK-LABEL: @generic_scan
// CHECK-NOT: llvm.intr.ctpop
tt.func private @generic_scan(%arg0: tensor<32xi32, #l32>) -> tensor<32xi32, #l32> {
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  tt.return %0 : tensor<32xi32, #l32>
}

// Keep the test functions from being DCE'd (mirrors scan_to_llvm.mlir).
tt.func public @anchor(%ptr: !llvm.ptr,
                       %ai1: !llvm.struct<(i1)>, %ai32: !llvm.struct<(i32)>) {
  %i1 = builtin.unrealized_conversion_cast %ai1 : !llvm.struct<(i1)> to tensor<32xi1, #l32>
  %i32 = builtin.unrealized_conversion_cast %ai32 : !llvm.struct<(i32)> to tensor<32xi32, #l32>

  %r0 = tt.call @bool_extui(%i1) : (tensor<32xi1, #l32>) -> tensor<32xi32, #l32>
  %r1 = tt.call @bool_and_mask(%i32) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  %r2 = tt.call @bool_select(%i1) : (tensor<32xi1, #l32>) -> tensor<32xi32, #l32>
  %r3 = tt.call @not_bool_extsi(%i1) : (tensor<32xi1, #l32>) -> tensor<32xi32, #l32>
  %r4 = tt.call @generic_scan(%i32) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  %r5 = tt.call @bool_reverse(%i1) : (tensor<32xi1, #l32>) -> tensor<32xi32, #l32>
  %r6 = tt.call @bool_extui_i64(%i1) : (tensor<32xi1, #l32>) -> tensor<32xi64, #l32>
  %r7 = tt.call @bool_i1(%i1) : (tensor<32xi1, #l32>) -> tensor<32xi1, #l32>

  %s0 = builtin.unrealized_conversion_cast %r0 : tensor<32xi32, #l32> to !llvm.struct<(i32)>
  llvm.store volatile %s0, %ptr : !llvm.struct<(i32)>, !llvm.ptr
  %s1 = builtin.unrealized_conversion_cast %r1 : tensor<32xi32, #l32> to !llvm.struct<(i32)>
  llvm.store volatile %s1, %ptr : !llvm.struct<(i32)>, !llvm.ptr
  %s2 = builtin.unrealized_conversion_cast %r2 : tensor<32xi32, #l32> to !llvm.struct<(i32)>
  llvm.store volatile %s2, %ptr : !llvm.struct<(i32)>, !llvm.ptr
  %s3 = builtin.unrealized_conversion_cast %r3 : tensor<32xi32, #l32> to !llvm.struct<(i32)>
  llvm.store volatile %s3, %ptr : !llvm.struct<(i32)>, !llvm.ptr
  %s4 = builtin.unrealized_conversion_cast %r4 : tensor<32xi32, #l32> to !llvm.struct<(i32)>
  llvm.store volatile %s4, %ptr : !llvm.struct<(i32)>, !llvm.ptr
  %s5 = builtin.unrealized_conversion_cast %r5 : tensor<32xi32, #l32> to !llvm.struct<(i32)>
  llvm.store volatile %s5, %ptr : !llvm.struct<(i32)>, !llvm.ptr
  %s6 = builtin.unrealized_conversion_cast %r6 : tensor<32xi64, #l32> to !llvm.struct<(i64)>
  llvm.store volatile %s6, %ptr : !llvm.struct<(i64)>, !llvm.ptr
  %s7 = builtin.unrealized_conversion_cast %r7 : tensor<32xi1, #l32> to !llvm.struct<(i1)>
  llvm.store volatile %s7, %ptr : !llvm.struct<(i1)>, !llvm.ptr
  tt.return
}

}
