#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_CXTOUPPERHCZH

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
class CxToUpperHCzH final : public BaseMQSSPass<CxToUpperHCzH>,
                            public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CxToUpperHCzH)

  StringRef getArgument() const override { return "CxToUpperHCzH"; }

  StringRef getDescription() const override {
    return "Decomposition pass of two-qubits cnot by H, Cz, and H";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto cxOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!cxOp || cxOp.getControls().size() != 1 ||
          cxOp.getTargets().size() != 1) {
        return;
      }

      IRRewriter rewriter(cxOp->getContext());
      Value control = cxOp.getControls()[0];
      Value target = cxOp.getTargets()[0];
      Location loc = cxOp.getLoc();
      rewriter.setInsertionPointAfter(cxOp);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.create<quake::ZOp>(loc, control, target);
      rewriter.create<quake::HOp>(loc, target);
      rewriter.eraseOp(cxOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createCxToUpperHCzHPass() {
  return std::make_unique<CxToUpperHCzH>();
}