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
-------------------------------------------------------------------------
  author Martin Letras
  date   January 2025
  version 1.0
  brief
    MLIR/Quake pass that dump a given Quake circuit into Tikz for visualization
    purposes.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "Passes/CodeGen.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

#include <iomanip>
#include <iostream>
#include <regex>

using namespace mlir;
using namespace mqss::support::quakeDialect;

void dumpQuakeOperationToTikz(Operation *op,
                              std::vector<std::vector<std::string>> &qubitLines,
                              std::vector<int> &depths) {
  if (op->getDialect()->getNamespace() != "quake" || isa<quake::AllocaOp>(op) ||
      isa<quake::ExtractRefOp>(op)) {
    // Do nothing if it is not a quake operation.
    // do nothing if the next operation is a qubit allocation.
    return;
  }
  auto gateName = std::string(op->getName().getStringRef());
  if (size_t pos = gateName.find("quake."); pos != std::string::npos) {
    // 6 is the length of "quake"
    gateName.erase(pos, 6);
  }

  auto padTo = [&](int qubit, int upto) {
    while (static_cast<int>(qubitLines[qubit].size()) <= upto) {
      qubitLines[qubit].push_back("\\qw");
    }
  };

  std::vector<double> parameters;
  std::vector<int> targets;
  std::vector<int> controls;
  bool isAdj = false;
  if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)) {
    for (auto operand : op->getOperands()) {
      if (operand.getType().isa<quake::RefType>()) {
        auto qubitIndexOpt =
            extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
        if (!qubitIndexOpt.has_value()) {
          continue;
        }
        int qubitIndex = qubitIndexOpt.value();
        assert(qubitIndex != -1 && "Non valid qubit index for measurement!");
        targets.push_back(qubitIndex);
      } else {
        throw std::runtime_error("Vecotrized measurements not supported.");
      }
    }
  } else {
    auto gate = dyn_cast<quake::OperatorInterface>(op);
    parameters = getParametersValues(gate.getParameters());
    targets = getIndicesOfValueRange(gate.getTargets());
    controls = getIndicesOfValueRange(gate.getControls());
    isAdj = gate.isAdj();
  }

  std::vector<int> qubits_in_use = targets;
  qubits_in_use.insert(qubits_in_use.end(), controls.begin(), controls.end());
  if (qubits_in_use.empty()) {
    return;
  }

  int min_qubit = *std::ranges::min_element(qubits_in_use);
  int max_qubit = *std::ranges::max_element(qubits_in_use);
  int op_depth = 0;
  for (int qubit = min_qubit; qubit <= max_qubit; qubit++) {
    op_depth = std::max(op_depth, depths[qubit]);
  }
  // Measurements
  if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)) {
    for (int qubit : targets) {
      padTo(qubit, op_depth);
      qubitLines[qubit].push_back("\\meter{}");
      depths[qubit] = op_depth + 1;
    }
    return;
  }

  // SWAP
  if (isa<quake::SwapOp>(op)) {
    if (targets.size() != 2) {
      std::cerr << "Detected SWAP with not exactly two targets, dropping op."
                << std::endl;
      return;
    }
    int q0 = targets[0], q1 = targets[1];
    padTo(q0, op_depth);
    padTo(q1, op_depth);
    qubitLines[q0].push_back("\\swap{" + std::to_string(q1 - q0) + "}");
    qubitLines[q1].push_back("\\swap{}");
  } else {
    // Other gates
    if (targets.size() != 1) {
      std::cerr << "Detected gate with not exactly one target, dropping op."
                << std::endl;
      return;
    }
    for (int qubit : targets) {
      std::string labelGate = gateName;
      // Rename phased_rx -> r for readbility.
      labelGate = std::regex_replace(labelGate, std::regex("phased_rx"), "r");
      if (isAdj) {
        labelGate = labelGate + "\\textsuperscript{\\textdagger}";
      }
      if (parameters.size() > 0) {
        labelGate += "(";
      }
      int countParam = 0;
      for (double parameter : parameters) {
        std::ostringstream tmpOss;
        tmpOss << std::fixed << std::setprecision(2) << parameter;
        labelGate += std::string(tmpOss.str());
        if (countParam != parameters.size() - 1) {
          labelGate += ",";
        }
        countParam++;
      }
      if (parameters.size() > 0) {
        labelGate += ")";
      }
      padTo(qubit, op_depth);
      qubitLines[qubit].push_back("\\gate{" + labelGate + "}");
    }
  }

  for (int qubit : controls) {
    int target0 = targets[0] - qubit;
    padTo(qubit, op_depth);
    qubitLines[qubit].push_back("\\ctrl{" + std::to_string(target0) + "}");
  }
  for (int qubit = min_qubit; qubit <= max_qubit; qubit++) {
    // Fill between qubits with wires.
    depths[qubit] = op_depth + 1;
  }
}

namespace {

class QuakeToTikzPass final
    : public PassWrapper<QuakeToTikzPass, OperationPass<FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuakeToTikzPass)

  explicit QuakeToTikzPass(llvm::raw_string_ostream &ostream)
      : outputStream(ostream) {}

  StringRef getArgument() const override { return "convert-quake-to-tikz"; }

  StringRef getDescription() const override {
    return "Lower Quake Operations to LaTeX TiKz";
  };

  void runOnOperation() override {
    auto circuit = getOperation();
    // Get the function name
    if (StringRef funcName = circuit.getName();
        funcName.find(std::string(CUDAQ_PREFIX_FUNCTION)) ==
        std::string::npos) {
      // Do nothing if the function is not cudaq kernel
      return;
    }

    std::map<int, int> measurements;
    int numQubits = getNumberOfQubits(circuit);
    std::vector<std::vector<std::string>> qubitTikz(numQubits);
    for (int i = 0; i < numQubits; i++) {
      qubitTikz[i].push_back({"\\lstick{\\ket{0}}"});
    }
    std::vector depths(numQubits, 0);
    circuit.walk([&](Operation *op) {
      dumpQuakeOperationToTikz(op, qubitTikz, depths);
    });
    size_t width = 0;
    for (auto &row : qubitTikz) {
      width = std::max(width, row.size());
    }
    for (auto &row : qubitTikz) {
      while (row.size() < width) {
        row.push_back("\\qw");
      }
    }

    outputStream << "\\begin{quantikz}\n";
    int countQubit = 0;
    for (auto qubit : qubitTikz) {
      int size = qubit.size();
      int countOps = 0;
      outputStream << "\t";
      for (auto op : qubit) {
        outputStream << op;
        if (countOps != size - 1) {
          outputStream << "\t&\t";
        }
        countOps++;
      }
      if (countQubit != numQubits - 1) {
        outputStream << "\\\\\n";
      } else {
        outputStream << "\n";
      }
      countQubit++;
    }
    outputStream << "\\end{quantikz}";
  }

private:
  llvm::raw_string_ostream &outputStream; // Store the tikz circuit
};

} // namespace

std::unique_ptr<Pass>
mqss::opt::createQuakeToTikzPass(llvm::raw_string_ostream &ostream) {
  return std::make_unique<QuakeToTikzPass>(ostream);
}