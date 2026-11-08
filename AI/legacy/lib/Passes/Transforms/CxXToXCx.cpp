#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/CommutateOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CXXTOXCX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class CxXToXCx final : public BaseMQSSPass<CxXToXCx>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxXToXCx)

  StringRef getArgument() const override { return "CxXToXCx"; }

  StringRef getDescription() const override {
    return "Apply commutation pass to pattern CNot-X to X-CNot";
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
      auto optional_cxOp
          = getPreviousOperationOnTarget(xOp, xOp.getTargets()[0]);
      if (!optional_cxOp) {
        return;
      }
      auto cxOp = dyn_cast_or_null<quake::XOp>(*optional_cxOp);
      if (!cxOp
          || cxOp.getTargets().size() != 1
          || cxOp.getControls().size() != 1
          || cxOp.getTargets()[0] != xOp.getTargets()[0]) {
        return;
      }
      IRRewriter rewriter(xOp->getContext());
      rewriter.setInsertionPointAfter(xOp);
      ValueRange targets = cxOp.getTargets();
      ValueRange controls = cxOp.getControls();
      Location loc = cxOp.getLoc();
      rewriter.create<quake::XOp>(loc, false, targets);
      rewriter.create<quake::XOp>(loc, false, controls, targets);
      rewriter.eraseOp(cxOp);
      rewriter.eraseOp(xOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCxXToXCxPass() {
  return std::make_unique<CxXToXCx>();
}