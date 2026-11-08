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
#define GEN_PASS_DEF_SZTOSDG

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class SZToSdg final : public BaseMQSSPass<SZToSdg>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SZToSdg)

  StringRef getArgument() const override { return "SZToSdg"; }

  StringRef getDescription() const override {
    return "Replaces a pattern composed of S and Z by Sdg";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto zOp = dyn_cast_or_null<quake::ZOp>(*op);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()) {
        return;
      }
      auto optional_sOp
          = getPreviousOperationOnTarget(zOp, zOp.getTargets()[0]);
      if (!optional_sOp) {
        return;
      }
      auto sOp = dyn_cast_or_null<quake::SOp>(*optional_sOp);
      if (!sOp
          || sOp.isAdj()
          || sOp.getTargets().size() != 1
          || !sOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(zOp->getContext());
      rewriter.setInsertionPointAfter(zOp);
      Location loc = sOp.getLoc();
      ValueRange targets = sOp.getTargets();
      rewriter.create<quake::SOp>(loc, true, targets);
      rewriter.eraseOp(sOp);
      rewriter.eraseOp(zOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSZToSdgPass() {
  return std::make_unique<SZToSdg>();
}