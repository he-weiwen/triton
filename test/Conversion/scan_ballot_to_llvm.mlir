// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm 2>&1 | FileCheck %s

// A boolean (0/1) integer add-scan lowers the warp scan to ballot+popcount
// (llvm.intr.ctpop); anything not provably {0,1} keeps the generic shuffle scan
// (no ctpop). Path selection is purely structural -- a function of the operand's
// producer chain and the combine op -- so the cases are distinguished by their
// input IR alone, with no runtime toggle.

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

// A narrow (i8) boolean upcast for the scan uses `extsi` (narrow ints are
// signed). `extsi` of a proven-{0,1} value wider than i1 keeps it {0,1}, so this
// must still fire -- regression guard for the int8/int16 path.
// CHECK-LABEL: @bool_extsi_narrow
// CHECK: llvm.intr.ctpop
tt.func private @bool_extsi_narrow(%arg0: tensor<32xi8, #l32>) -> tensor<32xi32, #l32> {
  %c1 = arith.constant dense<1> : tensor<32xi8, #l32>
  %m = arith.andi %arg0, %c1 : tensor<32xi8, #l32>
  %b = arith.extsi %m : tensor<32xi8, #l32> to tensor<32xi32, #l32>
  %0 = "tt.scan"(%b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %c: i32):
    %1 = arith.addi %a, %c : i32
    tt.scan.return %1 : i32
  }) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  tt.return %0 : tensor<32xi32, #l32>
}

// `select` between {0,1} arms is {0,1}.
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

// MISCOMPILE TRAP: `extsi` of an i1 yields {0,-1} (the i1 "1" sign-extends to
// all-ones), NOT {0,1}, so the fast path must NOT fire here.
// CHECK-LABEL: @not_bool_extsi_i1
// CHECK-NOT: llvm.intr.ctpop
tt.func private @not_bool_extsi_i1(%arg0: tensor<32xi1, #l32>) -> tensor<32xi32, #l32> {
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
                       %ai1: !llvm.struct<(i1)>, %ai8: !llvm.struct<(i8)>,
                       %ai32: !llvm.struct<(i32)>) {
  %i1 = builtin.unrealized_conversion_cast %ai1 : !llvm.struct<(i1)> to tensor<32xi1, #l32>
  %i8 = builtin.unrealized_conversion_cast %ai8 : !llvm.struct<(i8)> to tensor<32xi8, #l32>
  %i32 = builtin.unrealized_conversion_cast %ai32 : !llvm.struct<(i32)> to tensor<32xi32, #l32>

  %r0 = tt.call @bool_extui(%i1) : (tensor<32xi1, #l32>) -> tensor<32xi32, #l32>
  %r1 = tt.call @bool_and_mask(%i32) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>
  %r2 = tt.call @bool_extsi_narrow(%i8) : (tensor<32xi8, #l32>) -> tensor<32xi32, #l32>
  %r3 = tt.call @bool_select(%i1) : (tensor<32xi1, #l32>) -> tensor<32xi32, #l32>
  %r4 = tt.call @not_bool_extsi_i1(%i1) : (tensor<32xi1, #l32>) -> tensor<32xi32, #l32>
  %r5 = tt.call @generic_scan(%i32) : (tensor<32xi32, #l32>) -> tensor<32xi32, #l32>

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
  tt.return
}

}
