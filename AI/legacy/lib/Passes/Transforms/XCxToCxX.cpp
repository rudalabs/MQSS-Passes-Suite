#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/Transforms/CommutateOperations.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_XCXTOCXX

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::transforms;

namespace {

class XCxToCxX final : public BaseMQSSPass<XCxToCxX>, public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(XCxToCxX)

  StringRef getArgument() const override { return "XCxToCxX"; }

  StringRef getDescription() const override {
    return "Apply commutation pass to pattern X-Cx to Cx-X";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    kernel.walk([&](Operation *op) {
      auto cxOp = dyn_cast_or_null<quake::XOp>(*op);
      if (!cxOp
          || cxOp.getTargets().size() != 1
          || cxOp.getControls().size() != 1) {
        return;
      }
      auto optional_xOp
          = getPreviousOperationOnTarget(cxOp, cxOp.getTargets()[0]);
      if (!optional_xOp) {
        return;
      }
      auto xOp = dyn_cast_or_null<quake::XOp>(*optional_xOp);
      if (!xOp
          || xOp.getTargets().size() != 1
          || !xOp.getControls().empty()) {
        return;
      }
      IRRewriter rewriter(cxOp->getContext());
      rewriter.setInsertionPointAfter(cxOp);
      ValueRange targets = xOp.getTargets();
      ValueRange controls = cxOp.getControls();
      Location loc = xOp.getLoc();
      rewriter.create<quake::XOp>(loc, false, controls, targets);
      rewriter.create<quake::XOp>(loc, false, targets);
      rewriter.eraseOp(xOp);
      rewriter.eraseOp(cxOp);
      this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createXCxToCxXPass() {
  return std::make_unique<XCxToCxX>();
}