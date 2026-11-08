#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_XXTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class XXToId final : public BaseMQSSPass<XXToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XXToId)

  StringRef getArgument() const override { return "XXToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive X gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto xOp2 = dyn_cast_or_null<quake::XOp>(*op);
      if (!xOp2 || xOp2.getTargets().size() != 1 ||
          !xOp2.getControls().empty()) {
        return;
      }
      auto optional_xOp1 =
          getPreviousOperationOnTarget(xOp2, xOp2.getTargets()[0]);
      if (!optional_xOp1) {
        return;
      }
      auto xOp1 = dyn_cast_or_null<quake::XOp>(*optional_xOp1);
      if (!xOp1 || xOp1.getTargets().size() != 1 ||
          !xOp1.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(xOp2->getContext());
      rewriter.eraseOp(xOp2);
      rewriter.eraseOp(xOp1);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createXXToIdPass() {
  return std::make_unique<XXToId>();
}