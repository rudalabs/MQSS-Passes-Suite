#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_REVERSECX

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

class ReverseCx final : public BaseMQSSPass<ReverseCx>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReverseCx)

  StringRef getArgument() const override { return "ReverseCx"; }

  StringRef getDescription() const override {
    return "Transforms CX(0,1) to H(0) H(1) CX(1,0) H(1) H(0)";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto cxOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!cxOp
          || cxOp.getControls().size() != 1
          || cxOp.getTargets().size() != 1) {
        return;
      }
      Value control = cxOp.getControls()[0];
      Value target = cxOp.getTargets()[0];
      Location loc = cxOp.getLoc();

      IRRewriter rewriter(cxOp->getContext());
      rewriter.setInsertionPointAfter(cxOp);
      rewriter.create<quake::HOp>(loc, control);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.create<quake::XOp>(loc, target, control);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.create<quake::HOp>(loc, control);
      rewriter.eraseOp(cxOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createReverseCxPass() {
  return std::make_unique<ReverseCx>();
}