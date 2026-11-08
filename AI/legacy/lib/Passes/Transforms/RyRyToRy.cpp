#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_RYRYTORY
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class RyRyToRy final : public BaseMQSSPass<RyRyToRy>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RyRyToRy)

  StringRef getArgument() const override { return "RyRyToRy"; }

  StringRef getDescription() const override {
    return "Collapse consecutive Ry gates";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto ryOp2 = dyn_cast_or_null<quake::RyOp>(*op);
      if (!ryOp2
          || ryOp2.getTargets().size() != 1
          || !ryOp2.getControls().empty()
          || ryOp2.getParameters().size() != 1) {
        return;
      }
      auto optional_ryOp1
          = getPreviousOperationOnTarget(ryOp2, ryOp2.getTargets()[0]);
      if (!optional_ryOp1) {
        return;
      }
      auto ryOp1 = dyn_cast_or_null<quake::RyOp>(*optional_ryOp1);
      if (!ryOp1
          || ryOp1.getTargets().size() != 1
          || !ryOp1.getControls().empty()
          || ryOp1.getParameters().size() != 1) {
        return;
      }
      auto ry1Params = getOperationParameters(ryOp1);
      auto ry2Params = getOperationParameters(ryOp2);
      if (ry1Params.size() != 1 || ry2Params.size() != 1) {
        return;
      }
      double angle = ry1Params[0] + ry2Params[0];
      IRRewriter rewriter(ryOp2->getContext());
      rewriter.setInsertionPointAfter(ryOp2);
      Location loc = ryOp1.getLoc();
      ValueRange targets = ryOp1.getTargets();
      Value params = createFloatValue(rewriter, loc, angle);
      rewriter.create<quake::RyOp>(loc, false, params, ValueRange{}, targets);
      rewriter.eraseOp(ryOp1);
      rewriter.eraseOp(ryOp2);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createRyRyToRyPass() {
  return std::make_unique<RyRyToRy>();
}