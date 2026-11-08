#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_ZTOHXH

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"
} // namespace mqss::opt

using namespace mlir;

namespace {
class ZToHXH final : public BaseMQSSPass<ZToHXH>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZToHXH)

  StringRef getArgument() const override { return "ZToHXH"; }

  StringRef getDescription() const override {
    return "Decomposition pass of Z by H, X and H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto zOp = dyn_cast_or_null<quake::ZOp>(*op);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(zOp->getContext());
      Value target = zOp.getTargets()[0];
      Location loc = zOp.getLoc();
      rewriter.setInsertionPointAfter(zOp);
      rewriter.create<quake::HOp>(loc, false, target);
      rewriter.create<quake::XOp>(loc, false, target);
      rewriter.create<quake::HOp>(loc, false, target);
      rewriter.eraseOp(zOp);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createZToHXHPass() {
  return std::make_unique<ZToHXH>();
}