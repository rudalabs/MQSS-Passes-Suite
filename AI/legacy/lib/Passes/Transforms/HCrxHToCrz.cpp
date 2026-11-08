#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_HCRXHTOCRZ

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class HCrxHToCrz final : public BaseMQSSPass<HCrxHToCrz>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCrxHToCrz)

  StringRef getArgument() const override { return "HCrxHToCrz"; }

  StringRef getDescription() const override {
    return "Fold H CRx H to CRz";
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
      auto optional_crxOp
          = getPreviousOperationOnTarget(hOp2, hOp2.getTargets()[0]);
      if (!optional_crxOp) {
        return;
      }
      auto crxOp = dyn_cast_or_null<quake::RxOp>(*optional_crxOp);
      if (!crxOp
          || crxOp.isAdj()
          || crxOp.getTargets().size() != 1
          || crxOp.getControls().size() != 1
          || crxOp.getParameters().size() != 1) {
        return;
      }
      auto optional_hOp1
          = getPreviousOperationOnTarget(crxOp, crxOp.getTargets()[0]);
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
      ValueRange targets = crxOp.getTargets();
      ValueRange controls = crxOp.getControls();
      ValueRange params = crxOp.getParameters();
      Location loc = crxOp.getLoc();
      rewriter.create<quake::RzOp>(loc, false, params, controls, targets);
      rewriter.eraseOp(hOp1);
      rewriter.eraseOp(crxOp);
      rewriter.eraseOp(hOp2);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHCrxHToCrzPass() {
  return std::make_unique<HCrxHToCrz>();
}