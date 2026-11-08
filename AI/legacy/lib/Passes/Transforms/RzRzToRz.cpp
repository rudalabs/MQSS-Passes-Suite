#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_RZRZTORZ

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class RzRzToRz final : public BaseMQSSPass<RzRzToRz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RzRzToRz)

  StringRef getArgument() const override { return "RzRzToRz"; }

  StringRef getDescription() const override {
    return "Collapse consecutive Rz gates";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto rzOp2 = dyn_cast_or_null<quake::RzOp>(*op);
      if (!rzOp2
          || rzOp2.getTargets().size() != 1
          || !rzOp2.getControls().empty()
          || rzOp2.getParameters().size() != 1) {
        return;
      }
      auto optional_rzOp1
          = getPreviousOperationOnTarget(rzOp2, rzOp2.getTargets()[0]);
      if (!optional_rzOp1) {
        return;
      }
      auto rzOp1 = dyn_cast_or_null<quake::RzOp>(*optional_rzOp1);
      if (!rzOp1
          || rzOp1.getTargets().size() != 1
          || !rzOp1.getControls().empty()
          || rzOp1.getParameters().size() != 1) {
        return;
      }
      auto rz1Params = getOperationParameters(rzOp1);
      auto rz2Params = getOperationParameters(rzOp2);
      if (rz1Params.size() != 1 || rz2Params.size() != 1) {
        return;
      }
      double angle = rz1Params[0] + rz2Params[0];
      IRRewriter rewriter(rzOp2->getContext());
      rewriter.setInsertionPointAfter(rzOp2);
      Location loc = rzOp1.getLoc();
      ValueRange targets = rzOp1.getTargets();
      Value params = createFloatValue(rewriter, loc, angle);
      rewriter.create<quake::RzOp>(loc, params, ValueRange{}, targets);
      rewriter.eraseOp(rzOp1);
      rewriter.eraseOp(rzOp2);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createRzRzToRzPass() {
  return std::make_unique<RzRzToRz>();
}