#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/mlir_utils.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_ZERORYTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class ZeroRyToId final : public BaseMQSSPass<ZeroRyToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroRyToId)

  StringRef getArgument() const override { return "ZeroRyToId"; }

  StringRef getDescription() const override {
    return "Optimization pass that removes Ry rotations with zero angles";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto ryOp = dyn_cast_or_null<quake::RyOp>(*op);
      if (!ryOp
          || ryOp.getTargets().size() != 1
          || !ryOp.getControls().empty()
          || ryOp.getParameters().size() != 1) {
        return;
      }

      std::vector<double> params = getOperationParameters(ryOp);
      if (params.size() != 1) {
        return;
      }
      if (isMultipleOfTwoPi(params[0])) {
        IRRewriter rewriter(ryOp->getContext());
        rewriter.eraseOp(ryOp);
        this->wasApplied->store(true);
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZeroRyToIdPass() {
  return std::make_unique<ZeroRyToId>();
}