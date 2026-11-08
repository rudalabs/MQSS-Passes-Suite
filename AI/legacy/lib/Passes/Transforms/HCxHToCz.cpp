#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_HCXHTOCZ

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class HCxHToCz final : public BaseMQSSPass<HCxHToCz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCxHToCz)

  StringRef getArgument() const override { return "HCxHToCz"; }

  StringRef getDescription() const override {
    return "Fold H CNOT H to CZ";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto hOp2 = dyn_cast_or_null<quake::HOp>(*op);
      if (!hOp2
          || hOp2.getTargets().size() != 1
          || !hOp2.getControls().empty()) {
        return;
      }
      auto optional_cxOp
          = getPreviousOperationOnTarget(hOp2, hOp2.getTargets()[0]);
      if (!optional_cxOp) {
        return;
      }
      auto cxOp = dyn_cast_or_null<quake::XOp>(*optional_cxOp);
      if (!cxOp
          || cxOp.getTargets().size() != 1
          || cxOp.getControls().size() != 1) {
        return;
      }
      auto optional_hOp1
          = getPreviousOperationOnTarget(cxOp, cxOp.getTargets()[0]);
      if (!optional_hOp1) {
        return;
      }
      auto hOp1 = dyn_cast_or_null<quake::HOp>(*optional_hOp1);
      if (!hOp1
          || hOp1.getTargets().size() != 1
          || !hOp1.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(hOp2->getContext());
      rewriter.setInsertionPointAfter(hOp2);
      ValueRange targets = cxOp.getTargets();
      ValueRange controls = cxOp.getControls();
      Location loc = cxOp.getLoc();
      rewriter.create<quake::ZOp>(loc, false, controls, targets);
      rewriter.eraseOp(hOp1);
      rewriter.eraseOp(cxOp);
      rewriter.eraseOp(hOp2);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHCxHToCzPass() {
  return std::make_unique<HCxHToCz>();
}