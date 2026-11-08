#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_XHZTOH

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class XHZToH final : public BaseMQSSPass<XHZToH>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XHZToH)

  StringRef getArgument() const override { return "XHZToH"; }

  StringRef getDescription() const override {
    return "Optimization pass that replaces a pattern X H Z by H";
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
      auto optional_hOp
          = getPreviousOperationOnTarget(zOp, zOp.getTargets()[0]);
      if (!optional_hOp) {
        return;
      }
      auto hOp = dyn_cast_or_null<quake::HOp>(*optional_hOp);
      if (!hOp
          || hOp.getTargets().size() != 1
          || !hOp.getControls().empty()) {
        return;
      }
      auto optional_xOp
          = getPreviousOperationOnTarget(hOp, hOp.getTargets()[0]);
      if (!optional_xOp) {
        return;
      }
      auto xOp = dyn_cast_or_null<quake::XOp>(*optional_xOp);
      if (!xOp
          || xOp.getTargets().size() != 1
          || !xOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(zOp->getContext());
      rewriter.setInsertionPointAfter(zOp);
      ValueRange targets = xOp.getTargets();
      Location loc = xOp.getLoc();
      rewriter.create<quake::HOp>(loc, false, targets);
      rewriter.eraseOp(xOp);
      rewriter.eraseOp(hOp);
      rewriter.eraseOp(zOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createXHZToHPass() {
  return std::make_unique<XHZToH>();
}