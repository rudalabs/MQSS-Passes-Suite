#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/CommutateOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CXRXTORXCX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class CxRxToRxCx final : public BaseMQSSPass<CxRxToRxCx>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxRxToRxCx)

  StringRef getArgument() const override { return "CxRxToRxCx"; }

  StringRef getDescription() const override {
    return "Apply commutation pass of pattern CNot-Rx to Rx-CNot";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto rxOp = dyn_cast_or_null<quake::RxOp>(*op);
      if (!rxOp
          || rxOp.isAdj()
          || rxOp.getTargets().size() != 1
          || !rxOp.getControls().empty()
          || rxOp.getParameters().size() != 1) {
        return;
      }
      auto optional_cxOp
          = getPreviousOperationOnTarget(rxOp, rxOp.getTargets()[0]);
      if (!optional_cxOp) {
        return;
      }
      auto cxOp = dyn_cast_or_null<quake::XOp>(*optional_cxOp);
      if (!cxOp
          || cxOp.getTargets().size() != 1
          || cxOp.getControls().size() != 1
          || cxOp.getTargets()[0] != rxOp.getTargets()[0]) {
        return;
      }
      IRRewriter rewriter(rxOp->getContext());
      rewriter.setInsertionPointAfter(rxOp);
      ValueRange targets = cxOp.getTargets();
      ValueRange controls = cxOp.getControls();
      ValueRange params = rxOp.getParameters();
      Location loc = cxOp.getLoc();
      rewriter.create<quake::RxOp>(loc, false, params, ValueRange{}, targets);
      rewriter.create<quake::XOp>(loc, false, ValueRange{}, controls, targets);
      rewriter.eraseOp(cxOp);
      rewriter.eraseOp(rxOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCxRxToRxCxPass() {
  return std::make_unique<CxRxToRxCx>();
}