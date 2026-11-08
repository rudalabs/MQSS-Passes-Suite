#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_RXRXTORX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class RxRxToRx final : public BaseMQSSPass<RxRxToRx>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RxRxToRx)

  StringRef getArgument() const override { return "RxRxToRx"; }

  StringRef getDescription() const override {
    return "Collapse consecutive Rx gates";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto rxOp2 = dyn_cast_or_null<quake::RxOp>(*op);
      if (!rxOp2
          || rxOp2.getTargets().size() != 1
          || !rxOp2.getControls().empty()
          || rxOp2.getParameters().size() != 1) {
        return;
      }
      auto optional_rxOp1
          = getPreviousOperationOnTarget(rxOp2, rxOp2.getTargets()[0]);
      if (!optional_rxOp1) {
        return;
      }
      auto rxOp1 = dyn_cast_or_null<quake::RxOp>(*optional_rxOp1);
      if (!rxOp1
          || rxOp1.getTargets().size() != 1
          || !rxOp1.getControls().empty()
          || rxOp1.getParameters().size() != 1) {
        return;
      }
      auto rx1Params = getOperationParameters(rxOp1);
      auto rx2Params = getOperationParameters(rxOp2);
      if (rx1Params.size() != 1 || rx2Params.size() != 1) {
        return;
      }
      double angle = rx1Params[0] + rx2Params[0];
      IRRewriter rewriter(rxOp2->getContext());
      rewriter.setInsertionPointAfter(rxOp2);
      Location loc = rxOp1.getLoc();
      ValueRange targets = rxOp1.getTargets();
      Value params = createFloatValue(rewriter, loc, angle);
      rewriter.create<quake::RxOp>(loc, params, ValueRange{}, targets);
      rewriter.eraseOp(rxOp1);
      rewriter.eraseOp(rxOp2);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createRxRxToRxPass() {
  return std::make_unique<RxRxToRx>();
}