#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_HCZHTOCX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class HCzHToCx final : public BaseMQSSPass<HCzHToCx>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCzHToCx)

  StringRef getArgument() const override { return "HCzHToCx"; }

  StringRef getDescription() const override {
    return "Fold H Cz H to Cx";
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
      auto optional_czOp
          = getPreviousOperationOnTarget(hOp2, hOp2.getTargets()[0]);
      if (!optional_czOp) {
        return;
      }
      auto czOp = dyn_cast_or_null<quake::ZOp>(*optional_czOp);
      if (!czOp
          || czOp.getTargets().size() != 1
          || czOp.getControls().size() != 1) {
        return;
      }
      auto optional_hOp1
          = getPreviousOperationOnTarget(czOp, czOp.getTargets()[0]);
      if (!optional_hOp1) {
        return;
      }
      auto hOp1 = dyn_cast_or_null<quake::HOp>(*optional_hOp1);
      if (!hOp1
          || hOp1.getTargets().size() != 1
          || !hOp1.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(hOp2->getContext());
      rewriter.setInsertionPointAfter(hOp2);
      ValueRange targets = czOp.getTargets();
      ValueRange controls = czOp.getControls();
      Location loc = czOp.getLoc();
      rewriter.create<quake::XOp>(loc, false, controls, targets);
      rewriter.eraseOp(hOp1);
      rewriter.eraseOp(czOp);
      rewriter.eraseOp(hOp2);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHCzHToCxPass() {
  return std::make_unique<HCzHToCx>();
}