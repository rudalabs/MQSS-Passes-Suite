#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_ZHXTOH

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class ZHXToH final : public BaseMQSSPass<ZHXToH>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHXToH)

  StringRef getArgument() const override { return "ZHXToH"; }

  StringRef getDescription() const override { return "Fold Z H X to H"; }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto xOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!xOp
          || xOp.getTargets().size() != 1
          || !xOp.getControls().empty()) {
        return;
      }
      auto optional_hOp
          = getPreviousOperationOnTarget(xOp, xOp.getTargets()[0]);
      if (!optional_hOp) {
        return;
      }
      auto hOp = dyn_cast_or_null<quake::HOp>(*optional_hOp);
      if (!hOp
          || hOp.getTargets().size() != 1
          || !hOp.getControls().empty()) {
        return;
      }
      auto optional_zOp
          = getPreviousOperationOnTarget(hOp, hOp.getTargets()[0]);
      if (!optional_zOp) {
        return;
      }
      auto zOp = dyn_cast_or_null<quake::ZOp>(*optional_zOp);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(xOp->getContext());
      rewriter.setInsertionPointAfter(xOp);
      Value target = zOp.getTargets()[0];
      Location loc = zOp.getLoc();
      rewriter.create<quake::HOp>(loc, false, target);
      rewriter.eraseOp(zOp);
      rewriter.eraseOp(hOp);
      rewriter.eraseOp(xOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZHXToHPass() {
  return std::make_unique<ZHXToH>();
}