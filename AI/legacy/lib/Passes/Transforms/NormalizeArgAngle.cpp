/* This code and any associated documentation is provided "as is"

Copyright 2024 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/Munich-Quantum-Software-Stack/passes/blob/develop/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*************************************************************************
  author Martin Letras
  date   January 2025
  version 1.0
*************************************************************************/

#include "Passes/BaseMQSSPass.hpp"
#include "Passes/Transforms.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>
#include <numbers>

// Include auto-generated pass registration
namespace mqss::opt {
#define GEN_PASS_DEF_NORMALIZEARGANGLE

// NOLINTNEXTLINE
#include "Passes/Transforms.h.inc"
} // namespace mqss::opt
using namespace mlir;
using namespace mqss::support::quakeDialect;

namespace {

bool normalizeAngleOfRotations(Operation *currentOp, OpBuilder builder) {
  if (!isa<quake::RxOp>(currentOp) && !isa<quake::RyOp>(currentOp) &&
      !isa<quake::RzOp>(currentOp))
    return false; // do nothing if it is not rotation
  auto gate = dyn_cast<quake::OperatorInterface>(currentOp);
  double pi = std::numbers::pi;
  std::vector<Value> nParameters = {};
  IRRewriter rewriter(gate->getContext());
  for (auto parameter : gate.getParameters()) {
    auto optional_param_value =
        extractDoubleArgumentValue(parameter.getDefiningOp());
    if (!optional_param_value.has_value()) {
      return false;
    }
    double param = optional_param_value.value();
    param = param - std::floor(param / (2 * pi)) * 2 * pi;
    nParameters.push_back(createFloatValue(builder, gate.getLoc(), param));
  }
  ValueRange normParameters(nParameters);
  rewriter.setInsertionPointAfter(gate);
  if (isa<quake::RxOp>(gate)) {
    rewriter.create<quake::RxOp>(gate.getLoc(), gate.isAdj(), normParameters,
                                 gate.getControls(), gate.getTargets());
  } else if (isa<quake::RyOp>(gate)) {
    rewriter.create<quake::RyOp>(gate.getLoc(), gate.isAdj(), normParameters,
                                 gate.getControls(), gate.getTargets());
  } else if (isa<quake::RzOp>(gate)) {
    rewriter.create<quake::RzOp>(gate.getLoc(), gate.isAdj(), normParameters,
                                 gate.getControls(), gate.getTargets());
  }
  rewriter.eraseOp(gate);
  return true;
}

class NormalizeArgAngle final : public BaseMQSSPass<NormalizeArgAngle>,
                                public AppliedCheckPass {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeArgAngle)

  StringRef getArgument() const override { return "NormalizeArgAngle"; }

  StringRef getDescription() const override {
    return "Normalizes the angle of Rx, Ry and Rz rotations";
  }

  void operationsOnQuantumKernel(FuncOp kernel) override {
    this->wasApplied->store(false);
    OpBuilder builder(&kernel.getBody());
    kernel.walk([&](Operation *op) {
      if (normalizeAngleOfRotations(op, builder))
        this->wasApplied->store(true);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mqss::opt::createNormalizeArgAnglePass() {
  return std::make_unique<NormalizeArgAngle>();
}