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
#define GEN_PASS_DEF_HZHTOX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"

} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class HZHToX final : public BaseMQSSPass<HZHToX>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HZHToX)

  StringRef getArgument() const override { return "HZHToX"; }

  StringRef getDescription() const override {
    return "Optimization pass that replaces a pattern composed of H, Z, H by X";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto hOp2 = dyn_cast_or_null<quake::HOp>(*op);
      if (!hOp2
          || hOp2.getTargets().size() != 1
          || !hOp2.getControls().empty()) {
        return;
      }
      auto optional_zOp
          = getPreviousOperationOnTarget(hOp2, hOp2.getTargets()[0]);
      if (!optional_zOp) {
        return;
      }
      auto zOp = dyn_cast_or_null<quake::ZOp>(*optional_zOp);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()) {
        return;
      }
      auto optional_hOp1
          = getPreviousOperationOnTarget(zOp, zOp.getTargets()[0]);
      if (!optional_hOp1) {
        return;
      }
      auto hOp1 = dyn_cast_or_null<quake::HOp>(*optional_hOp1);
      if (!hOp1
          || hOp1.getTargets().size() != 1
          || !hOp1.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(hOp2->getContext());
      rewriter.setInsertionPointAfter(hOp2);
      ValueRange targets = hOp1.getTargets();
      Location loc = hOp1.getLoc();
      rewriter.create<quake::XOp>(loc, false, targets);
      rewriter.eraseOp(hOp1);
      rewriter.eraseOp(zOp);
      rewriter.eraseOp(hOp2);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHZHToXPass() {
  return std::make_unique<HZHToX>();
}