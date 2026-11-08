#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_HXTOZH

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class HXToZH final : public BaseMQSSPass<HXToZH>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HXToZH)

  StringRef getArgument() const override { return "HXToZH"; }

  StringRef getDescription() const override {
    return "Transforms H X to Z H";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto xOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!xOp
          || xOp.getTargets().size() != 1
          || !xOp.getControls().empty()) {
        return;
      }
      auto optional_hOp
          = getPreviousOperationOnTarget(xOp, xOp.getTargets()[0]);
      if (!optional_hOp) {
        return;
      }
      auto hOp = dyn_cast_or_null<quake::HOp>(*optional_hOp);
      if (!hOp
          || hOp.getTargets().size() != 1
          || !hOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(xOp->getContext());
      rewriter.setInsertionPointAfter(xOp);
      ValueRange targets = hOp.getTargets();
      Location loc = hOp.getLoc();
      rewriter.create<quake::ZOp>(loc, false, targets);
      rewriter.create<quake::HOp>(loc, false, targets);
      rewriter.eraseOp(hOp);
      rewriter.eraseOp(xOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHXToZHPass() {
  return std::make_unique<HXToZH>();
}