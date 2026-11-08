#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CZTOLOWERHCXH

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
class CzToLowerHCxH final : public BaseMQSSPass<CzToLowerHCxH>,
                            public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CzToLowerHCxH)

  StringRef getArgument() const override { return "CzToLowerHCxH"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Cz by H, Cx, and H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto czOp = dyn_cast_or_null<quake::ZOp>(*op);
      if (!czOp || czOp.getControls().size() != 1 ||
          czOp.getTargets().size() != 1) {
        return;
      }

      IRRewriter rewriter(czOp->getContext());
      Value control = czOp.getControls()[0];
      Value target = czOp.getTargets()[0];
      Location loc = czOp.getLoc();
      rewriter.setInsertionPointAfter(czOp);
      rewriter.create<quake::HOp>(loc, control);
      rewriter.create<quake::XOp>(loc, target, control);
      rewriter.create<quake::HOp>(loc, control);
      rewriter.eraseOp(czOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCzToLowerHCxHPass() {
  return std::make_unique<CzToLowerHCxH>();
}