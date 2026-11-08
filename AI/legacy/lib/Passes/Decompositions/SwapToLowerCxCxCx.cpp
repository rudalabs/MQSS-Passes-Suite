#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_SWAPTOLOWERCXCXCX

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
class SwapToLowerCxCxCx final : public BaseMQSSPass<SwapToLowerCxCxCx>,
                                public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwapToLowerCxCxCx)

  StringRef getArgument() const override { return "SwapToLowerCxCxCx"; }

  StringRef getDescription() const override {
    return "Decompose SWAP by CX(1,0) CX(0,1) CX(1,0)";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto swapOp = dyn_cast_or_null<quake::SwapOp>(*op);
      if (!swapOp || swapOp.getTargets().size() != 2 ||
          !swapOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(swapOp->getContext());
      Value q0 = swapOp.getTargets()[0];
      Value q1 = swapOp.getTargets()[1];
      Location loc = swapOp.getLoc();
      rewriter.setInsertionPointAfter(swapOp);
      rewriter.create<quake::XOp>(loc, q1, q0);
      rewriter.create<quake::XOp>(loc, q0, q1);
      rewriter.create<quake::XOp>(loc, q1, q0);
      rewriter.eraseOp(swapOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSwapToLowerCxCxCxPass() {
  return std::make_unique<SwapToLowerCxCxCx>();
}