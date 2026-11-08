#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/mlir_utils.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_TDGTTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class TdgTToId final : public BaseMQSSPass<TdgTToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TdgTToId)

  StringRef getArgument() const override { return "TdgTToId"; }

  StringRef getDescription() const override {
    return "Remove Tdg followed by T.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto tOp2 = dyn_cast_or_null<quake::TOp>(*op);
      if (!tOp2
          || tOp2.isAdj()
          || tOp2.getTargets().size() != 1
          || !tOp2.getControls().empty()) {
        return;
      }
      auto optional_tOp1 =
          getPreviousOperationOnTarget(tOp2, tOp2.getTargets()[0]);
      if (!optional_tOp1) {
        return;
      }
      auto tOp1 = dyn_cast_or_null<quake::TOp>(*optional_tOp1);
      if (!tOp1
          || !tOp1.isAdj()
          || tOp1.getTargets().size() != 1
          || !tOp1.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(tOp2->getContext());
      rewriter.eraseOp(tOp2);
      rewriter.eraseOp(tOp1);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createTdgTToIdPass() {
  return std::make_unique<TdgTToId>();
}