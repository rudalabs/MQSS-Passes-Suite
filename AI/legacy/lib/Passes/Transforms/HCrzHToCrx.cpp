#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_HCRZHTOCRX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class HCrzHToCrx final : public BaseMQSSPass<HCrzHToCrx>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HCrzHToCrx)

  StringRef getArgument() const override { return "HCrzHToCrx"; }

  StringRef getDescription() const override {
    return "Fold H CRz H to CRx";
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
      auto optional_crzOp
          = getPreviousOperationOnTarget(hOp2, hOp2.getTargets()[0]);
      if (!optional_crzOp) {
        return;
      }
      auto crzOp = dyn_cast_or_null<quake::RzOp>(*optional_crzOp);
      if (!crzOp
          || crzOp.isAdj()
          || crzOp.getTargets().size() != 1
          || crzOp.getControls().size() != 1
          || crzOp.getParameters().size() != 1) {
        return;
      }
      auto optional_hOp1
          = getPreviousOperationOnTarget(crzOp, crzOp.getTargets()[0]);
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
      ValueRange targets = hOp1.getTargets();
      ValueRange controls = crzOp.getControls();
      ValueRange params = crzOp.getParameters();
      Location loc = hOp1.getLoc();
      rewriter.create<quake::RxOp>(loc, false, params, controls, targets);
      rewriter.eraseOp(hOp1);
      rewriter.eraseOp(crzOp);
      rewriter.eraseOp(hOp2);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHCrzHToCrxPass() {
  return std::make_unique<HCrzHToCrx>();
}