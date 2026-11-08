#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_HHTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class HHToId final : public BaseMQSSPass<HHToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HHToId)

  StringRef getArgument() const override { return "HHToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive H gates.";
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
      auto optional_hOp1
          = getPreviousOperationOnTarget(hOp2, hOp2.getTargets()[0]);
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
      rewriter.eraseOp(hOp2);
      rewriter.eraseOp(hOp1);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createHHToIdPass() {
  return std::make_unique<HHToId>();
}