#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_CXCXCXTOSWAP

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::quakeDialect;

namespace {

class CxCxCxToSwap final : public BaseMQSSPass<CxCxCxToSwap>,
                           public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxCxCxToSwap)

  StringRef getArgument() const override { return "CxCxCxToSwap"; }

  StringRef getDescription() const override {
    return "Replace CNOT CNOT CNOT swap pattern by SWAP";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto cxOp3 = dyn_cast_or_null<quake::XOp>(*op);
      if (!cxOp3 || cxOp3.getTargets().size() != 1 ||
          cxOp3.getControls().size() != 1) {
        return;
      }
      auto optional_cxOp2_onTarget =
          getPreviousOperationOnTarget(cxOp3, cxOp3.getTargets()[0]);
      auto optional_cxOp2_onControl =
          getPreviousOperationOnTarget(cxOp3, cxOp3.getControls()[0]);
      if (!optional_cxOp2_onTarget || !optional_cxOp2_onControl ||
          optional_cxOp2_onTarget != optional_cxOp2_onControl) {
        return;
      }
      auto cxOp2 = dyn_cast_or_null<quake::XOp>(*optional_cxOp2_onTarget);
      if (!cxOp2 || cxOp2.getTargets().size() != 1 ||
          cxOp2.getControls().size() != 1 ||
          cxOp2.getControls()[0] != cxOp3.getTargets()[0] ||
          cxOp2.getTargets()[0] != cxOp3.getControls()[0]) {
        return;
      }
      auto optional_cxOp1_onTarget =
          getPreviousOperationOnTarget(cxOp2, cxOp2.getTargets()[0]);
      auto optional_cxOp1_onControl =
          getPreviousOperationOnTarget(cxOp2, cxOp2.getControls()[0]);
      if (!optional_cxOp1_onTarget || !optional_cxOp1_onControl ||
          optional_cxOp1_onTarget != optional_cxOp1_onControl) {
        return;
      }
      auto cxOp1 = dyn_cast_or_null<quake::XOp>(*optional_cxOp1_onTarget);
      if (!cxOp1 || cxOp1.getTargets().size() != 1 ||
          cxOp1.getControls().size() != 1 ||
          cxOp2.getControls()[0] != cxOp1.getTargets()[0] ||
          cxOp2.getTargets()[0] != cxOp1.getControls()[0] ||
          cxOp3.getControls()[0] != cxOp1.getControls()[0] ||
          cxOp3.getTargets()[0] != cxOp1.getTargets()[0]) {
        return;
      }
      IRRewriter rewriter(cxOp3->getContext());
      rewriter.setInsertionPointAfter(cxOp3);
      ValueRange targets{cxOp1.getControls()[0], cxOp1.getTargets()[0]};
      Location loc = cxOp1.getLoc();
      rewriter.create<quake::SwapOp>(loc, targets);
      rewriter.eraseOp(cxOp1);
      rewriter.eraseOp(cxOp2);
      rewriter.eraseOp(cxOp3);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCxCxCxToSwapPass() {
  return std::make_unique<CxCxCxToSwap>();
}