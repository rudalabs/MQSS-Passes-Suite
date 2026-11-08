#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_RZTOHRXH

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
class RzToHRxH final : public BaseMQSSPass<RzToHRxH>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RzToHRxH)

  StringRef getArgument() const override { return "RzToHRxH"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Rz by H, Rx, and H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto rzOp = dyn_cast_or_null<quake::RzOp>(*op);
      if (!rzOp
          || rzOp.isAdj()
          || rzOp.getTargets().size() != 1
          || !rzOp.getControls().empty()
          || rzOp.getParameters().size() != 1) {
        return;
      }

      IRRewriter rewriter(rzOp->getContext());
      Value target = rzOp.getTargets()[0];
      Value param = rzOp.getParameters()[0];
      Location loc = rzOp.getLoc();
      rewriter.setInsertionPointAfter(rzOp);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.create<quake::RxOp>(loc, false, param, ValueRange{}, target);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.eraseOp(rzOp);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createRzToHRxHPass() {
  return std::make_unique<RzToHRxH>();
}