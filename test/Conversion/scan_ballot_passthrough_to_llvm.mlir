// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm 2>&1 | FileCheck %s

// convert_layout is the one data-movement passthrough the {0,1} proof looks
// through. It preserves element type and only relayouts its operand, so it
// carries {0,1}-ness to the scan operand and the boolean add-scan still lowers
// to ballot+popcount (llvm.intr.ctpop). It is kept -- unlike the other movement
// ops (splat/broadcast/expand_dims/reshape/trans), which fold away before
// lowering or never reach a real scan operand -- because a genuine relayout can
// legitimately survive directly on the scan operand at TTGIR->LLVM time, where
// convert folding is suppressed. A convert_layout of a non-{0,1} value must NOT
// fire: the proof recurses into the relayouted `src`.

#b2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#b2c = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// convert_layout: {0,1} produced in one layout, relayouted into the scan layout.
// CHECK-LABEL: @convert_layout_2d
// CHECK: llvm.intr.ctpop
tt.func private @convert_layout_2d(%arg0: tensor<4x32xi1, #b2c>) -> tensor<4x32xi32, #b2> {
  %e = arith.extui %arg0 : tensor<4x32xi1, #b2c> to tensor<4x32xi32, #b2c>
  %c = ttg.convert_layout %e : tensor<4x32xi32, #b2c> -> tensor<4x32xi32, #b2>
  %0 = "tt.scan"(%c) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%a: i32, %d: i32):
    %1 = arith.addi %a, %d : i32
    tt.scan.return %1 : i32
  }) : (tensor<4x32xi32, #b2>) -> tensor<4x32xi32, #b2>
  tt.return %0 : tensor<4x32xi32, #b2>
}

// GUARD: a convert_layout of an arbitrary (non-{0,1}) value must NOT fire -- the
// arm recurses into `src`, so relayouting a generic i32 keeps the generic scan.
// (Last scan func; CHECK-NOT runs to EOF.)
// CHECK-LABEL: @not_bool_convert_layout
// CHECK-NOT: llvm.intr.ctpop
tt.func private @not_bool_convert_layout(%arg0: tensor<4x32xi32, #b2c>) -> tensor<4x32xi32, #b2> {
  %c = ttg.convert_layout %arg0 : tensor<4x32xi32, #b2c> -> tensor<4x32xi32, #b2>
  %0 = "tt.scan"(%c) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%a: i32, %d: i32):
    %1 = arith.addi %a, %d : i32
    tt.scan.return %1 : i32
  }) : (tensor<4x32xi32, #b2>) -> tensor<4x32xi32, #b2>
  tt.return %0 : tensor<4x32xi32, #b2>
}

}
