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
  date   February 2025
  version 1.0
  brief
    MLIR/Quake pass that converts an input QASM circuit to Quake.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "Interfaces/QASMToQuake.hpp"
#include "Passes/CodeGen.hpp"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "ir/parsers/qasm3_parser/Parser.hpp"
#include "ir/parsers/qasm3_parser/Statement.hpp"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <iomanip>
#include <ranges>
#include <regex>
#include <unordered_map>

using namespace mqss::interfaces;
using mlir::OperationPass;
using mlir::Pass;
using mlir::PassWrapper;
using mlir::StringRef;
using mlir::Type;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;

namespace {
class QASM3ToQuake final
    : public PassWrapper<QASM3ToQuake, OperationPass<FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QASM3ToQuake)

  QASM3ToQuake(std::istringstream &qasmStream, bool measureAllQubits)
      : qasmStream(qasmStream), measureAllQubits(measureAllQubits) {}

  StringRef getArgument() const override { return "convert-qasm3-to-quake"; }

  StringRef getDescription() const override {
    return "Convert QASM3 to Quake Operations";
  }

  void runOnOperation() override {
    auto circuit = getOperation();
    // Get the function name
    if (StringRef funcName = circuit.getName();
        funcName.find(std::string(CUDAQ_PREFIX_FUNCTION)) == std::string::npos)
      return; // do nothing if the function is not cudaq kernel
    // Create the parser
    qasm3::Parser parser(&qasmStream, true);
    // Parse the program to get the AST
    std::vector<std::shared_ptr<qasm3::Statement>> program;
    // Parse the program
    try {
      program = parser.parseProgram();
    } catch (const std::runtime_error &e) {
      llvm::outs() << "Parsing failed: " << e.what() << "\n";
      assert(false && "Error!");
    }
    // First, I do need to know the place of the return operation
    // then every new inserted operation will be before the "return" statement
    Operation *returnOp;
    circuit.walk([&](Operation *op) {
      if (isa<ReturnOp>(op)) {
        // Check if it's a return op
        returnOp = op;
      }
    });
    assert(returnOp && "Error: No return operation found!\n");
    // I declare a single Builder that can be used by the different parsing
    // process!
    OpBuilder builder(circuit.getContext());
    Location loc = circuit.getLoc();
    // Traverse the AST
    auto [allocatedQubitVectors, orderVectors] =
        insertAllocatedQubits(program, builder, loc, returnOp);
    if (allocatedQubitVectors.size() == 0) {
      return; // if no allocated qubits return nothing
    }
    // for debugging print maps of qubits
#ifdef DEBUG
    for (const auto &vector : allocatedQubitVectors | std::views::keys) {
      llvm::outs() << "QASM vector " << vector << "\n";
    }
#endif
    // Parse and insert gates
    for (const auto &statement : program)
      // Check if the statement is a GateCallStatement
      if (auto gateCall =
              std::dynamic_pointer_cast<qasm3::GateCallStatement>(statement)) {
        insertGate(gateCall, builder, loc, returnOp, allocatedQubitVectors);
      }
#ifdef DEBUG
    llvm::outs() << "Gates were inserted!\n";
#endif
    if (measureAllQubits) {
      // Apply measurements in all allocated qubit vectors
      builder.setInsertionPoint(returnOp);
      Type measTy = quake::MeasureType::get(builder.getContext());
      for (const auto &key : orderVectors | std::views::keys) {
        auto stdVectType = cudaq::cc::StdvecType::get(measTy);
        builder.create<quake::MzOp>(loc, stdVectType,
                                    allocatedQubitVectors.at(key));
      }
    } else {
      parseAndInsertMeasurements(program, builder, loc, returnOp,
                                 allocatedQubitVectors);
    }
  }

private:
  std::istringstream &qasmStream;
  bool measureAllQubits = false;
};

} // namespace

std::unique_ptr<Pass>
mqss::opt::createQASM3ToQuakePass(std::istringstream &qasmStream,
                                  bool measureAllQubits) {
  return std::make_unique<QASM3ToQuake>(qasmStream, measureAllQubits);
}