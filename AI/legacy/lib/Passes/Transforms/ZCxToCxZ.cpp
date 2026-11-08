#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_ZCXTOCXZ

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;


namespace {
class ZCxToCxZ final : public BaseMQSSPass<ZCxToCxZ>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZCxToCxZ)

  StringRef getArgument() const override { return "ZCxToCxZ"; }

  StringRef getDescription() const override {
    return "Commute Z(0) Cx(0,1) to Cx(0,1) Z(0)";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto cxOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!cxOp
          || cxOp.getTargets().size() != 1
          || cxOp.getControls().size() != 1) {
        return;
      }
      auto optional_zOp
          = getPreviousOperationOnTarget(cxOp, cxOp.getControls()[0]);
      if (!optional_zOp) {
        return;
      }
      auto zOp = dyn_cast_or_null<quake::ZOp>(*optional_zOp);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()
          || cxOp.getControls()[0] != zOp.getTargets()[0]) {
        return;
      }
      IRRewriter rewriter(cxOp->getContext());
      rewriter.setInsertionPointAfter(cxOp);
      ValueRange targets = cxOp.getTargets();
      ValueRange controls = cxOp.getControls();
      Location loc = cxOp.getLoc();
      rewriter.create<quake::XOp>(loc, false, controls, targets);
      rewriter.create<quake::ZOp>(loc, false, controls);
      rewriter.eraseOp(zOp);
      rewriter.eraseOp(cxOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZCxToCxZPass() {
  return std::make_unique<ZCxToCxZ>();
}