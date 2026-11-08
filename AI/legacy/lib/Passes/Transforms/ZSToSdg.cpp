#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_ZSTOSDG

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class ZSToSdg final : public BaseMQSSPass<ZSToSdg>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZSToSdg)

  StringRef getArgument() const override { return "ZSToSdg"; }

  StringRef getDescription() const override {
    return "Replaces a pattern composed of S Z by Sdg";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto sOp = dyn_cast_or_null<quake::SOp>(*op);
      if (!sOp
          || sOp.isAdj()
          || sOp.getTargets().size() != 1
          || !sOp.getControls().empty()) {
        return;
      }
      auto optional_zOp
          = getPreviousOperationOnTarget(sOp, sOp.getTargets()[0]);
      if (!optional_zOp) {
        return;
      }
      auto zOp = dyn_cast_or_null<quake::ZOp>(*optional_zOp);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(sOp->getContext());
      rewriter.setInsertionPointAfter(sOp);
      Value target = zOp.getTargets()[0];
      Location loc = zOp.getLoc();
      rewriter.create<quake::SOp>(loc, true, target);
      rewriter.eraseOp(zOp);
      rewriter.eraseOp(sOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZSToSdgPass() {
  return std::make_unique<ZSToSdg>();
}