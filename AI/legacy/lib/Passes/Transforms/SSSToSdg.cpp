#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "Support/Transforms/CommutateOperations.hpp"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_SSSTOSDG

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class SSSToSDG final : public BaseMQSSPass<SSSToSDG>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SSSToSDG)

  StringRef getArgument() const override { return "SSSToSDG"; }

  StringRef getDescription() const override {
    return "Replace S S S by SDG";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto sOp3 = dyn_cast_or_null<quake::SOp>(*op);
      if (!sOp3
          || sOp3.isAdj()
          || sOp3.getTargets().size() != 1
          || !sOp3.getControls().empty()) {
        return;
      }
      auto optional_sOp2
          = getPreviousOperationOnTarget(sOp3, sOp3.getTargets()[0]);
      if (!optional_sOp2) {
        return;
      }
      auto sOp2 = dyn_cast_or_null<quake::SOp>(*optional_sOp2);
      if (!sOp2
          || sOp2.isAdj()
          || sOp2.getTargets().size() != 1
          || !sOp2.getControls().empty()) {
        return;
      }
      auto optional_sOp1
          = getPreviousOperationOnTarget(sOp2, sOp2.getTargets()[0]);
      if (!optional_sOp1) {
        return;
      }
      auto sOp1 = dyn_cast_or_null<quake::SOp>(*optional_sOp1);
      if (!sOp1
          || sOp1.isAdj()
          || sOp1.getTargets().size() != 1
          || !sOp1.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(sOp3->getContext());
      rewriter.setInsertionPointAfter(sOp1);
      Location loc = sOp1.getLoc();
      ValueRange targets = sOp1.getTargets();
      rewriter.create<quake::SOp>(loc, true, targets);
      rewriter.eraseOp(sOp1);
      rewriter.eraseOp(sOp2);
      rewriter.eraseOp(sOp3);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createSSSToSdgPass() {
  return std::make_unique<SSSToSDG>();
}