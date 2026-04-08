#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPREFERREDCLUSTERFALLBACKPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

static bool hasCrossCTAConvertLayout(ttg::ConvertLayoutOp cvt) {
  auto kBlock = StringAttr::get(cvt->getContext(), "block");
  auto conversion = minimalCvtLayout(cvt.getSrc().getType(), cvt.getType());
  return conversion.hasInDim(kBlock);
}

static bool hasCrossCTAReduce(triton::ReduceOp reduce) {
  auto srcTy = reduce.getInputTypes()[0];
  auto splitNum = ttg::getCTASplitNum(srcTy.getEncoding());
  return splitNum[reduce.getAxis()] > 1;
}

static bool moduleRequestsConSan(ModuleOp mod) {
  for (StringRef attrName :
       {"ttg.instrumentation_mode", "triton.instrumentation_mode"}) {
    auto attr = mod->getAttrOfType<StringAttr>(attrName);
    if (attr && attr.getValue().contains("consan"))
      return true;
  }
  return false;
}

static bool isSupportedMBarrierType(ttg::MemDescType barrierTy) {
  auto kBlock = StringAttr::get(barrierTy.getContext(), "block");
  uint32_t cgaBroadcastMask =
      toLinearLayout(barrierTy).getFreeVariableMasks().lookup(kBlock);

  // Broadcast mbarriers use another CTA's barrier, so we only allow broadcast
  // on the first bit (i.e., CTA0 and CTA1).
  return cgaBroadcastMask <= 1;
}

class TritonNvidiaGPUPreferredClusterFallbackPass
    : public impl::TritonNvidiaGPUPreferredClusterFallbackPassBase<
          TritonNvidiaGPUPreferredClusterFallbackPass> {
public:
  using impl::TritonNvidiaGPUPreferredClusterFallbackPassBase<
      TritonNvidiaGPUPreferredClusterFallbackPass>::
      TritonNvidiaGPUPreferredClusterFallbackPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod->removeAttr(AttrPreferredClusterFallbackCTAsName);

    int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
    if (computeCapability < 100 || numCTAs <= 2)
      return;

    if (moduleRequestsConSan(mod))
      return;

    WalkResult result = mod.walk([&](Operation *op) -> WalkResult {
      auto unsupported = [&] { return WalkResult::interrupt(); };

      if (isa<triton::ElementwiseInlineAsmOp>(op))
        return unsupported();

      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
        if (hasCrossCTAConvertLayout(cvt))
          return unsupported();
        return WalkResult::advance();
      }

      if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
        if (hasCrossCTAReduce(reduce))
          return unsupported();
        return WalkResult::advance();
      }

      if (isa<ttng::AsyncTMAReduceOp, ttng::AsyncTMAGatherOp,
              ttng::AsyncTMAScatterOp>(op))
        return unsupported();

      if (isa<ttng::ClusterArriveOp, ttng::ClusterWaitOp,
              ttng::ClusterBarrierOp>(op))
        return unsupported();

      if (auto barrierOp = dyn_cast<ttg::MBarrierOpInterface>(op)) {
        for (Value barrier : barrierOp.getBarriers()) {
          auto barrierTy = cast<ttg::MemDescType>(barrier.getType());
          if (!isSupportedMBarrierType(barrierTy))
            return unsupported();
        }
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return;

    mod->setAttr(AttrPreferredClusterFallbackCTAsName,
                 IntegerAttr::get(IntegerType::get(mod.getContext(), 32), 2));
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
