#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_HYTOYH

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class HYToYH final : public BaseMQSSPass<HYToYH>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HYToYH)
  StringRef getArgument() const override { return "HYToYH"; }

  StringRef getDescription() const override {
    return "Pass that switches a pattern composed Hadamard and Y to Y and "
        "Hadamard";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto yOp = dyn_cast_or_null<quake::YOp>(*op);
      if (!yOp
          || yOp.getTargets().size() != 1
          || !yOp.getControls().empty()) {
        return;
      }
      auto optional_hOp
          = getPreviousOperationOnTarget(yOp, yOp.getTargets()[0]);
      if (!optional_hOp) {
        return;
      }
      auto hOp = dyn_cast_or_null<quake::HOp>(*optional_hOp);
      if (!hOp
          || hOp.getTargets().size() != 1
          || !hOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(yOp->getContext());
      rewriter.setInsertionPointAfter(yOp);
      ValueRange targets = hOp.getTargets();
      Location loc = hOp.getLoc();
      rewriter.create<quake::YOp>(loc, false, targets);
      rewriter.create<quake::HOp>(loc, false, targets);
      rewriter.eraseOp(hOp);
      rewriter.eraseOp(yOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHYToYHPass() {
  return std::make_unique<HYToYH>();
}