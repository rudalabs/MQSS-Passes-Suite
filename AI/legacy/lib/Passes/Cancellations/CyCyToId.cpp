#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CYCYTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class CyCyToId final : public BaseMQSSPass<CyCyToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CyCyToId)

  StringRef getArgument() const override { return "CyCyToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive identical Cy gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto cyOp2 = dyn_cast_or_null<quake::YOp>(*op);
      if (!cyOp2 || cyOp2.getTargets().size() != 1 ||
          cyOp2.getControls().size() != 1) {
        return;
      }
      auto optional_cyOp1_onTarget =
          getPreviousOperationOnTarget(cyOp2, cyOp2.getTargets()[0]);
      auto optional_cyOp1_onControl =
          getPreviousOperationOnTarget(cyOp2, cyOp2.getControls()[0]);
      if (!optional_cyOp1_onTarget || !optional_cyOp1_onControl ||
          optional_cyOp1_onTarget != optional_cyOp1_onControl) {
        return;
      }
      auto cyOp1 = dyn_cast_or_null<quake::YOp>(*optional_cyOp1_onTarget);
      if (!cyOp1 || cyOp1.getTargets().size() != 1 ||
          cyOp1.getControls().size() != 1 ||
          cyOp1.getTargets()[0] != cyOp2.getTargets()[0] ||
          cyOp1.getControls()[0] != cyOp2.getControls()[0]) {
        return;
      }

      IRRewriter rewriter(cyOp2->getContext());
      rewriter.eraseOp(cyOp2);
      rewriter.eraseOp(cyOp1);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCyCyToIdPass() {
  return std::make_unique<CyCyToId>();
}