#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/CommutateOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_RXCXTOCXRX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class RxCxToCxRx final : public BaseMQSSPass<RxCxToCxRx>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RxCxToCxRx)

  StringRef getArgument() const override { return "RxCxToCxRx"; }

  StringRef getDescription() const override {
    return "Apply commutation pass to pattern Rx-CNot to CNot-Rx";
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
      auto optional_rxOp
          = getPreviousOperationOnTarget(cxOp, cxOp.getTargets()[0]);
      if (!optional_rxOp) {
        return;
      }
      auto rxOp = dyn_cast_or_null<quake::RxOp>(*optional_rxOp);
      if (!rxOp
          || rxOp.getTargets().size() != 1
          || !rxOp.getControls().empty()
          || rxOp.getParameters().size() != 1) {
        return;
      }
      IRRewriter rewriter(cxOp->getContext());
      rewriter.setInsertionPointAfter(cxOp);
      ValueRange targets = cxOp.getTargets();
      ValueRange controls = cxOp.getControls();
      ValueRange parameters = rxOp.getParameters();
      Location loc = rxOp.getLoc();
      rewriter.create<quake::XOp>(loc, false, controls, targets);
      rewriter.create<quake::RxOp>(loc, parameters, ValueRange{}, targets);
      rewriter.eraseOp(rxOp);
      rewriter.eraseOp(cxOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createRxCxToCxRxPass() {
  return std::make_unique<RxCxToCxRx>();
}