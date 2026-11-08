#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CZCZTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class CzCzToId final : public BaseMQSSPass<CzCzToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CzCzToId)

  StringRef getArgument() const override { return "CzCzToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive identical CNOT gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto czOp2 = dyn_cast_or_null<quake::ZOp>(*op);
      if (!czOp2 || czOp2.getTargets().size() != 1 ||
          czOp2.getControls().size() != 1) {
        return;
      }
      auto optional_czOp1_onTarget =
          getPreviousOperationOnTarget(czOp2, czOp2.getTargets()[0]);
      auto optional_czOp1_onControl =
          getPreviousOperationOnTarget(czOp2, czOp2.getControls()[0]);
      if (!optional_czOp1_onTarget || !optional_czOp1_onControl ||
          optional_czOp1_onTarget != optional_czOp1_onControl) {
        return;
      }
      auto czOp1 = dyn_cast_or_null<quake::ZOp>(*optional_czOp1_onTarget);
      if (!czOp1 || czOp1.getTargets().size() != 1 ||
          czOp1.getControls().size() != 1 ||
          czOp1.getTargets()[0] != czOp2.getTargets()[0] ||
          czOp1.getControls()[0] != czOp2.getControls()[0]) {
        return;
      }

      IRRewriter rewriter(czOp2->getContext());
      rewriter.eraseOp(czOp2);
      rewriter.eraseOp(czOp1);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCzCzToIdPass() {
  return std::make_unique<CzCzToId>();
}