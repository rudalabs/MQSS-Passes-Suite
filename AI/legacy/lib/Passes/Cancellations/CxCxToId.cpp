#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

#include <mlir_utils.hpp>

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CXCXTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class CxCxToId final : public BaseMQSSPass<CxCxToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxCxToId)

  StringRef getArgument() const override { return "CxCxToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive identical Cx gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto cxOp2 = dyn_cast_or_null<quake::XOp>(*op);
      if (!cxOp2 || cxOp2.getTargets().size() != 1 ||
          cxOp2.getControls().size() != 1) {
        return;
      }
      auto optional_cxOp1_onTarget =
          getPreviousOperationOnTarget(cxOp2, cxOp2.getTargets()[0]);
      auto optional_cxOp1_onControl =
          getPreviousOperationOnTarget(cxOp2, cxOp2.getControls()[0]);
      if (!optional_cxOp1_onTarget || !optional_cxOp1_onControl ||
          optional_cxOp1_onTarget != optional_cxOp1_onControl) {
        return;
      }
      auto cxOp1 = dyn_cast_or_null<quake::XOp>(*optional_cxOp1_onTarget);
      if (!cxOp1 || cxOp1.getTargets().size() != 1 ||
          cxOp1.getControls().size() != 1 ||
          cxOp1.getTargets()[0] != cxOp2.getTargets()[0] ||
          cxOp1.getControls()[0] != cxOp2.getControls()[0]) {
        return;
      }

      IRRewriter rewriter(cxOp2->getContext());
      rewriter.eraseOp(cxOp2);
      rewriter.eraseOp(cxOp1);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCxCxToIdPass() {
  return std::make_unique<CxCxToId>();
}