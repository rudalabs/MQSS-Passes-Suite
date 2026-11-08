#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_STOSDGSDGSDG

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
class SToSdgSdgSdg final : public BaseMQSSPass<SToSdgSdgSdg>,
                           public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SToSdgSdgSdg)

  StringRef getArgument() const override { return "SToSdgSdgSdg"; }

  StringRef getDescription() const override {
    return "Decompoe S gate by Sdg Sdg Sdg";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto sOp = dyn_cast_or_null<quake::SOp>(*op);
      if (!sOp || sOp.isAdj() || sOp.getTargets().size() != 1 ||
          !sOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(sOp->getContext());
      Value target = sOp.getTargets()[0];
      Location loc = sOp.getLoc();
      rewriter.setInsertionPointAfter(sOp);
      rewriter.create<quake::SOp>(loc, true, target);
      rewriter.create<quake::SOp>(loc, true, target);
      rewriter.create<quake::SOp>(loc, true, target);
      rewriter.eraseOp(sOp);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createSToSdgSdgSdgPass() {
  return std::make_unique<SToSdgSdgSdg>();
}