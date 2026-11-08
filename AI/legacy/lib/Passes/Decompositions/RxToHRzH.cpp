#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_RXTOHRZH

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {

class RxToHRzH final : public BaseMQSSPass<RxToHRzH>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RxToHRzH)

  StringRef getArgument() const override { return "RxToHRzH"; }

  StringRef getDescription() const override {
    return "Decomposition pass that replaces Rx by H, Rz and H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto rxOp = dyn_cast_or_null<quake::RxOp>(*op);
      if (!rxOp
          || rxOp.isAdj()
          || rxOp.getTargets().size() != 1
          || !rxOp.getControls().empty()
          || rxOp.getParameters().size() != 1) {
        return;
      }

      IRRewriter rewriter(rxOp->getContext());
      Value target = rxOp.getTargets()[0];
      Value param = rxOp.getParameters()[0];
      Location loc = rxOp.getLoc();
      rewriter.setInsertionPointAfter(rxOp);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.create<quake::RzOp>(loc, false, param, ValueRange{}, target);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.eraseOp(rxOp);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createRxToHRzHPass() {
  return std::make_unique<RxToHRzH>();
}