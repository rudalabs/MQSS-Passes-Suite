#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/SwitchOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_ZHTOHX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class ZHToHX final : public BaseMQSSPass<ZHToHX>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHToHX)

  StringRef getArgument() const override { return "ZHToHX"; }

  StringRef getDescription() const override {
    return "Pass that switches a pattern composed by Z and Hadamard to "
        "Hadamard and X";
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
      auto optional_zOp
          = getPreviousOperationOnTarget(hOp, hOp.getTargets()[0]);
      if (!optional_zOp) {
        return;
      }
      auto zOp = dyn_cast_or_null<quake::ZOp>(*optional_zOp);
      if (!zOp
          || zOp.getTargets().size() != 1
          || !zOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(hOp->getContext());
      rewriter.setInsertionPointAfter(hOp);
      Value target = zOp.getTargets()[0];
      Location loc = zOp.getLoc();
      rewriter.create<quake::HOp>(loc, false, target);
      rewriter.create<quake::XOp>(loc, false, target);
      rewriter.eraseOp(zOp);
      rewriter.eraseOp(hOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZHToHXPass() {
  return std::make_unique<ZHToHX>();
}