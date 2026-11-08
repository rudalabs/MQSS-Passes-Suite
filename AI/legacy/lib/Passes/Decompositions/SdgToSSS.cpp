#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Decompositions.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_SDGTOSSS

// NOLINTNEXTLINE
#include "Passes/Decompositions.h.inc"

} // namespace mqss::opt
using namespace mlir;

namespace {
class SdgToSSS final : public BaseMQSSPass<SdgToSSS>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdgToSSS)

  StringRef getArgument() const override { return "SdgToSSS"; }

  StringRef getDescription() const override {
    return "Decomposition pass that replaces a Sdg gate by three S gates";
  }

  void operationsOnQuantumKernel(func::FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto sOp = dyn_cast_or_null<quake::SOp>(*op);
      if (!sOp
          || !sOp.isAdj()
          || sOp.getTargets().size() != 1
          || !sOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(sOp->getContext());
      Value target = sOp.getTargets()[0];
      Location loc = sOp.getLoc();
      rewriter.setInsertionPointAfter(sOp);
      rewriter.create<quake::SOp>(loc, false, target);
      rewriter.create<quake::SOp>(loc, false, target);
      rewriter.create<quake::SOp>(loc, false, target);
      rewriter.eraseOp(sOp);
      this->wasApplied->store(true);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mqss::opt::createSdgToSSSPass() {
  return std::make_unique<SdgToSSS>();
}