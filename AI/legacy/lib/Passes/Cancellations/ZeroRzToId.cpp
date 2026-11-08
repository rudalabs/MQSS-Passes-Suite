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
#define GEN_PASS_DEF_ZERORZTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class ZeroRzToId final : public BaseMQSSPass<ZeroRzToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroRzToId)

  StringRef getArgument() const override { return "ZeroRzToId"; }

  StringRef getDescription() const override {
    return "Optimization pass that removes Rz rotations with zero angles";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto rzOp = dyn_cast_or_null<quake::RzOp>(*op);
      if (!rzOp
          || rzOp.getTargets().size() != 1
          || !rzOp.getControls().empty()
          || rzOp.getParameters().size() != 1) {
        return;
      }

      std::vector<double> params = getOperationParameters(rzOp);
      if (params.size() != 1) {
        return;
      }
      if (isMultipleOfTwoPi(params[0])) {
        IRRewriter rewriter(rzOp->getContext());
        rewriter.eraseOp(rzOp);
        this->wasApplied->store(true);
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZeroRzToIdPass() {
  return std::make_unique<ZeroRzToId>();
}