#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Cancellations.hpp"
#include "Support/Transforms/CancellationOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mqss::opt {
#define GEN_PASS_DEF_ZZTOID

// NOLINTNEXTLINE
#include "Passes/Cancellations.h.inc"
} // namespace mqss::opt

using namespace mlir;
using namespace mqss::support::transforms;

namespace {
class ZZToId final : public BaseMQSSPass<ZZToId>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZZToId)

  StringRef getArgument() const override { return "ZZToId"; }

  StringRef getDescription() const override {
    return "Remove consecutive Z gates.";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto zOp2 = dyn_cast_or_null<quake::ZOp>(*op);
      if (!zOp2
          || zOp2.getTargets().size() != 1
          || !zOp2.getControls().empty()) {
        return;
      }
      auto optional_zOp1
          = getPreviousOperationOnTarget(zOp2, zOp2.getTargets()[0]);
      if (!optional_zOp1) {
        return;
      }
      auto zOp1 = dyn_cast_or_null<quake::ZOp>(*optional_zOp1);
      if (!zOp1
          || zOp1.getTargets().size() != 1
          || !zOp1.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(zOp2->getContext());
      rewriter.eraseOp(zOp2);
      rewriter.eraseOp(zOp1);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createZZToIdPass() {
  return std::make_unique<ZZToId>();
}