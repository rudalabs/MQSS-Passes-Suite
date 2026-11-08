#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_YYTOID
// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class YYToId final : public BaseMQSSPass<YYToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(YYToId)

  StringRef getArgument() const override { return "YYToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive Y gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto yOp2 = dyn_cast_or_null<quake::YOp>(*op);
      if (!yOp2
          || yOp2.getTargets().size() != 1
          || !yOp2.getControls().empty()) {
        return;
      }
      auto optional_yOp1
          = getPreviousOperationOnTarget(yOp2, yOp2.getTargets()[0]);
      if (!optional_yOp1) {
        return;
      }
      auto yOp1 = dyn_cast_or_null<quake::YOp>(*optional_yOp1);
      if (!yOp1
          || yOp1.getTargets().size() != 1
          || !yOp1.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(yOp2->getContext());
      rewriter.eraseOp(yOp2);
      rewriter.eraseOp(yOp1);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createYYToIdPass() {
  return std::make_unique<YYToId>();
}