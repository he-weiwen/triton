#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUGATHERTOLOCALGATHER
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUGatherToLocalGatherPass
    : public impl::TritonGPUGatherToLocalGatherBase<
          TritonGPUGatherToLocalGatherPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    auto result = getOperation().walk([&](GatherOp op) -> WalkResult {
      if (GatherLoweringHelper(op).isWarpLocal())
        return WalkResult::advance();

      Value src = op.getSrc();
      auto srcType = op.getSrc().getType();
      if (hoist)
        rewriter.setInsertionPointAfterValue(src);
      else
        rewriter.setInsertionPoint(op);
      if (srcType.getElementType().isInteger(1)) {
        srcType = srcType.clone(rewriter.getI8Type());
        src = arith::ExtUIOp::create(rewriter, op.getLoc(), srcType, src);
      } else if (isa<PointerType>(srcType.getElementType())) {
        srcType = srcType.clone(rewriter.getI64Type());
        src = PtrToIntOp::create(rewriter, op.getLoc(), srcType, src);
      }
      auto sharedEncoding = SwizzledSharedEncodingAttr::get(
          &getContext(), 1, 1, 1, getOrderForMemory(srcType),
          getCGALayout(srcType.getEncoding()));
      auto memDescType = MemDescType::get(
          srcType.getShape(), srcType.getElementType(), sharedEncoding,
          SharedMemorySpaceAttr::get(&getContext()));
      if (isCrossCTAGatherScatter(memDescType, op.getType(), op.getAxis())) {
        op.emitError("cross-CTA gathers are not supported");
        return WalkResult::interrupt();
      }
      auto alloc =
          LocalAllocOp::create(rewriter, op.getLoc(), memDescType, src);

      rewriter.setInsertionPoint(op);
      Value result = LocalGatherOp::create(
          rewriter, op.getLoc(), op.getType().clone(srcType.getElementType()),
          alloc, op.getIndices(), op.getAxisAttr());
      if (isa<PointerType>(op.getType().getElementType()))
        result =
            IntToPtrOp::create(rewriter, op.getLoc(), op.getType(), result);
      else if (result.getType() != op.getType())
        result = arith::TruncIOp::create(rewriter, op.getLoc(), op.getType(),
                                         result);
      rewriter.replaceOp(op, result);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
