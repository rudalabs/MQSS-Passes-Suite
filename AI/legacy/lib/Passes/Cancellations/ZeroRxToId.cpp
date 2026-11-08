#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_ZERORXTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class ZeroRxToId final : public BaseMQSSPass<ZeroRxToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroRxToId)

  StringRef getArgument() const override { return "ZeroRxToId"; }

  StringRef getDescription() const override {
    return "Optimization pass that removes Rx rotations with zero angles";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto rxOp = dyn_cast_or_null<quake::RxOp>(*op);
      if (!rxOp || rxOp.getTargets().size() != 1 ||
          !rxOp.getControls().empty() || rxOp.getParameters().size() != 1) {
        return;
      }

      std::vector<double> params = getOperationParameters(rxOp);
      if (params.size() != 1) {
        return;
      }
      if (isMultipleOfTwoPi(params[0])) {
        IRRewriter rewriter(rxOp->getContext());
        rewriter.eraseOp(rxOp);
        this->wasApplied->store(true);
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZeroRxToIdPass() {
  return std::make_unique<ZeroRxToId>();
}