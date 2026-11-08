#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_HZTOXH

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class HZToXH final : public BaseMQSSPass<HZToXH>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HZToXH)

  StringRef getArgument() const override { return "HZToXH"; }

  StringRef getDescription() const override {
    return "Pass that switches a pattern composed H Z to X and H";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto zOp = dyn_cast_or_null<quake::ZOp>(*op);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()) {
        return;
      }
      auto optional_hOp
          = getPreviousOperationOnTarget(zOp, zOp.getTargets()[0]);
      if (!optional_hOp) {
        return;
      }
      auto hOp = dyn_cast_or_null<quake::HOp>(*optional_hOp);
      if (!hOp
          || hOp.getTargets().size() != 1
          || !hOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(zOp->getContext());
      rewriter.setInsertionPointAfter(zOp);
      ValueRange targets = hOp.getTargets();
      Location loc = hOp.getLoc();
      rewriter.create<quake::XOp>(loc, false, targets);
      rewriter.create<quake::HOp>(loc, false, targets);
      rewriter.eraseOp(hOp);
      rewriter.eraseOp(zOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHZToXHPass() {
  return std::make_unique<HZToXH>();
}