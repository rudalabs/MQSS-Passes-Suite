#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_XTOHZH

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"
} // namespace mqss::opt

using namespace mlir;

namespace {
class XToHZH final : public BaseMQSSPass<XToHZH>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XToHZH)

  StringRef getArgument() const override { return "XToHZH"; }

  StringRef getDescription() const override {
    return "Decomposition pass of X by H, Z and H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto xOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!xOp
          || xOp.getTargets().size() != 1
          || !xOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(xOp->getContext());
      Value target = xOp.getTargets()[0];
      Location loc = xOp.getLoc();
      rewriter.setInsertionPointAfter(xOp);
      rewriter.create<quake::HOp>(loc, false, target);
      rewriter.create<quake::ZOp>(loc, false, target);
      rewriter.create<quake::HOp>(loc, false, target);
      rewriter.eraseOp(xOp);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createXToHZHPass() {
  return std::make_unique<XToHZH>();
}