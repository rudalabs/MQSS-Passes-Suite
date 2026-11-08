#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CRXTOHCRZH

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
class CrxToHCrzH final : public BaseMQSSPass<CrxToHCrzH>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CrxToHCrzH)

  StringRef getArgument() const override { return "CrxToHCrzH"; }

  StringRef getDescription() const override {
    return "Decomposition pass of crx by h, crz and h";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto crxOp = dyn_cast_or_null<quake::RxOp>(*op);
      if (!crxOp
          || crxOp.isAdj()
          || crxOp.getTargets().size() != 1
          || crxOp.getControls().size() != 1
          || crxOp.getParameters().size() != 1) {
        return;
      }

      IRRewriter rewriter(crxOp->getContext());
      Value control = crxOp.getControls()[0];
      Value target = crxOp.getTargets()[0];
      Value param = crxOp.getParameters()[0];
      Location loc = crxOp.getLoc();
      rewriter.setInsertionPointAfter(crxOp);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.create<quake::RzOp>(loc, false, param, control, target);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.eraseOp(crxOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCrxToHCrzHPass() {
  return std::make_unique<CrxToHCrzH>();
}