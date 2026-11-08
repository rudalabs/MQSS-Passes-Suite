#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_XHTOHZ

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class XHToHZ final : public BaseMQSSPass<XHToHZ>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XHToHZ)

  StringRef getArgument() const override { return "XHToHZ"; }

  StringRef getDescription() const override {
    return "Switches a pattern composed by X H to H Z";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto hOp = dyn_cast_or_null<quake::HOp>(*op);
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
      IRRewriter rewriter(hOp->getContext());
      rewriter.setInsertionPointAfter(hOp);
      ValueRange targets = xOp.getTargets();
      Location loc = xOp.getLoc();
      rewriter.create<quake::HOp>(loc, false, targets);
      rewriter.create<quake::ZOp>(loc, false, targets);
      rewriter.eraseOp(xOp);
      rewriter.eraseOp(hOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createXHToHZPass() {
  return std::make_unique<XHToHZ>();
}