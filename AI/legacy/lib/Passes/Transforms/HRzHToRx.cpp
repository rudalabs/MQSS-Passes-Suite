#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_HRZHTORX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class HRzHToRx final : public BaseMQSSPass<HRzHToRx>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HRzHToRx)

  StringRef getArgument() const override { return "HRzHToRx"; }

  StringRef getDescription() const override {
    return "Fold H Rz H to Rx";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto hOp2 = dyn_cast_or_null<quake::HOp>(*op);
      if (!hOp2
          || hOp2.getTargets().size() != 1
          || !hOp2.getControls().empty()) {
        return;
      }
      auto optional_rzOp
          = getPreviousOperationOnTarget(hOp2, hOp2.getTargets()[0]);
      if (!optional_rzOp) {
        return;
      }
      auto rzOp = dyn_cast_or_null<quake::RzOp>(*optional_rzOp);
      if (!rzOp
          || rzOp.isAdj()
          || rzOp.getTargets().size() != 1
          || !rzOp.getControls().empty()
          || rzOp.getParameters().size() != 1) {
        return;
      }
      auto optional_hOp1
          = getPreviousOperationOnTarget(rzOp, rzOp.getTargets()[0]);
      if (!optional_hOp1) {
        return;
      }
      auto hOp1 = dyn_cast_or_null<quake::HOp>(*optional_hOp1);
      if (!hOp1
          || hOp1.getTargets().size() != 1
          || !hOp1.getControls().empty()
          || hOp2.getTargets()[0] != hOp1.getTargets()[0]) {
        return;
      }
      IRRewriter rewriter(hOp2->getContext());
      rewriter.setInsertionPointAfter(hOp2);
      ValueRange targets = hOp1.getTargets();
      ValueRange params = rzOp.getParameters();
      Location loc = hOp1.getLoc();
      rewriter.create<quake::RxOp>(loc, false, params, ValueRange{}, targets);
      rewriter.eraseOp(hOp1);
      rewriter.eraseOp(rzOp);
      rewriter.eraseOp(hOp2);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHRzHToRxPass() {
  return std::make_unique<HRzHToRx>();
}