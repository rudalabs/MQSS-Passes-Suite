#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_YHTOHY

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class YHToHY final : public BaseMQSSPass<YHToHY>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(YHToHY)
  StringRef getArgument() const override { return "YHToHY"; }

  StringRef getDescription() const override {
    return "Switches a pattern composed by Y H to H Y";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto hOp = dyn_cast_or_null<quake::HOp>(*op);
      if (!hOp
          || hOp.getTargets().size() != 1
          || !hOp.getControls().empty()) {
        return;
      }
      auto optional_yOp
          = getPreviousOperationOnTarget(hOp, hOp.getTargets()[0]);
      if (!optional_yOp) {
        return;
      }
      auto yOp = dyn_cast_or_null<quake::YOp>(*optional_yOp);
      if (!yOp
          || yOp.getTargets().size() != 1
          || !yOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(hOp->getContext());
      rewriter.setInsertionPointAfter(hOp);
      Value target = yOp.getTargets()[0];
      Location loc = yOp.getLoc();
      rewriter.create<quake::HOp>(loc, false, target);
      rewriter.create<quake::YOp>(loc, false, target);
      rewriter.eraseOp(yOp);
      rewriter.eraseOp(hOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createYHToHYPass() {
  return std::make_unique<YHToHY>();
}