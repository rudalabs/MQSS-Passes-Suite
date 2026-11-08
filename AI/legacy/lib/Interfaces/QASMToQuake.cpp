/* This code and any associated documentation is provided "as is"

Copyright 2025 Munich Quantum Software Stack Project

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
    Functions used by the QASMToQuake Parser.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "Interfaces/QASMToQuake.hpp"

#include "Interfaces/Constants.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <iomanip>
#include <ranges>
#include <regex>
#include <unordered_map>

////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::Location;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::SmallVector;
using mlir::Type;
using mlir::Value;
////////////////////////////////////////////////////////////////////////////////

using namespace mqss::support::quakeDialect;

// Function to determine if a gate is a multi-qubit gate with implicit controls
bool mqss::interfaces::isMultiQubitGate(const std::string &gate) {
  std::string gatename = gate;
  std::ranges::transform(gatename, gatename.begin(),
                         [](unsigned char c) { return std::tolower(c); });
  unsigned int i = 0;
  while (i < gatename.size() && gatename[i] == 'r') {
    i++;
  }
  return i < gatename.size() && gatename[i] == 'c';
}

// Function to get the number of controls for a gate
size_t mqss::interfaces::getNumControls(const std::string &gate) {
  std::string gatename = gate;
  std::ranges::transform(gatename, gatename.begin(),
                         [](unsigned char c) { return std::tolower(c); });
  unsigned int i = 0;
  unsigned int n = 0;
  while (i < gatename.size() && gatename[i] == 'r') {
    i++;
  }
  while (i < gatename.size() && gatename[i] == 'c') {
    i++;
    n++;
  }
  return n;
}

// This function returns the set of quantum registers declared in a given QASM
// program.
std::tuple<QASMVectorToQuakeVector, std::vector<std::pair<std::string, int>>>
mqss::interfaces::insertAllocatedQubits(
    const std::vector<std::shared_ptr<qasm3::Statement>> &program,
    OpBuilder &builder, Location loc, Operation *inOp) {
  // I have to do it like this to preserve the order they are declared
  std::vector<std::pair<std::string, int>> orderVectors = {};
  int totalQubits = 0;
  for (const auto &statement : program) {
    // Check if the statement is a DeclarationStatement
    if (auto declStmt =
            std::dynamic_pointer_cast<qasm3::DeclarationStatement>(statement)) {
      //        std::cout << "type name " <<  typeid(*statement).name()  <<
      //        "\n";
#ifdef DEBUG
      std::cout << "identifier " << declStmt->identifier << "\n";
      std::cout << "expression " << declStmt->expression << "\n";
#endif
      // Checking the type contained in the variant
      auto &variantType = declStmt->type;
      if (auto designatedPtr = std::get_if<
              std::shared_ptr<qasm3::Type<std::shared_ptr<qasm3::Expression>>>>(
              &variantType)) {
        // If we successfully got the Type<std::shared_ptr<Expression>>, handle
        // it
        std::shared_ptr<qasm3::Type<std::shared_ptr<qasm3::Expression>>>
            typeExpr = *designatedPtr;
        // std::cout << "Type expression to string " << typeExpr->toString() <<
        // "\n"; std::cout << "Successfully cast to
        // Type<std::shared_ptr<Expression>>!" << std::endl;
        if (std::regex pattern("qubit");
            !std::regex_search(typeExpr->toString(), pattern)) {
          continue; // error code
        }
        if (auto designator = typeExpr->getDesignator()) {
          if (auto constant =
                  std::dynamic_pointer_cast<qasm3::Constant>(designator)) {
            orderVectors.push_back(std::make_pair(
                std::string(declStmt->identifier), constant->getSInt()));
            totalQubits += constant->getSInt(); // Access the variant
            // std::cout << "Total qubits: " << val << std::endl;
          }
        }
      }
    }
  }
  // identifier, qasmqubit, mlirqubit
  QASMVectorToQuakeVector mlirQubitVectors;
  // IDQASMMLIR mlirQubits;
  if (totalQubits == 0 || totalQubits == -1)
    return std::make_tuple(mlirQubitVectors,
                           orderVectors); // do nothing and return empty map
  // instead of returning do the insertion of the measurement in the MLIR module
  // return totalQubits;
  builder.setInsertionPoint(inOp); // Set insertion before return
  // create the different mlir vectors in the QASM program
  for (const auto &[order, totalQubits] : orderVectors) {
#ifdef DEBUG
    std::cout << "order " << order << "\n";
#endif
    // Define the type for a vector of totalQubits qubits
    auto qubitVecType = quake::VeqType::get(builder.getContext(), totalQubits);
    auto qubitReg = builder.create<quake::AllocaOp>(loc, qubitVecType);
    mlirQubitVectors.emplace(order, qubitReg);
  }
  return std::make_tuple(mlirQubitVectors, orderVectors);
}

double mqss::interfaces::evaluateExpression(
    const std::shared_ptr<qasm3::Expression> &expr) {
  if (auto constantExpr = std::dynamic_pointer_cast<qasm3::Constant>(expr)) {
    double val;
    if (constantExpr->isInt() || constantExpr->isSInt() ||
        constantExpr->isUInt())
      val = constantExpr->getSInt();
    else
      val = constantExpr->getFP();
    return val;
  }
  if (auto unaryExpr =
          std::dynamic_pointer_cast<qasm3::UnaryExpression>(expr)) {
    // Handle unary expressions like -pi
    double operandValue = evaluateExpression(unaryExpr->operand);
    switch (unaryExpr->op) {
    case qasm3::UnaryExpression::Op::Negate:
      return -operandValue;
    // Add other unary operations if needed
    default:
      assert(false && "Unsupported unary operation");
    }
  }
  if (auto binaryExpr =
          std::dynamic_pointer_cast<qasm3::BinaryExpression>(expr)) {
    // Handle binary expressions like pi/2
    double lhsValue = evaluateExpression(binaryExpr->lhs);
    double rhsValue = evaluateExpression(binaryExpr->rhs);
    switch (binaryExpr->op) {
    case qasm3::BinaryExpression::Op::Add:
      return lhsValue + rhsValue;
    case qasm3::BinaryExpression::Op::Subtract:
      return lhsValue - rhsValue;
    case qasm3::BinaryExpression::Op::Multiply:
      return lhsValue * rhsValue;
    case qasm3::BinaryExpression::Op::Divide:
      return lhsValue / rhsValue;
    // Add other binary operations if needed
    default:
      assert(false && "Unsupported binary operation");
    }
  }
  if (auto identifierExpr =
          std::dynamic_pointer_cast<qasm3::IdentifierExpression>(expr)) {
    // Handle identifiers like pi
    if (identifierExpr->identifier == "pi") {
      return PI; // Use the value of pi from <cmath>
    }
    assert(false &&
           ("Unsupported identifier: " + identifierExpr->identifier).c_str());
  }
  assert(false && "Unsupported expression type");
}

// Function that inserts a QASM gate into a MLIR/Quake module
void mqss::interfaces::insertGate(
    const std::shared_ptr<qasm3::GateCallStatement> &gateCall,
    OpBuilder &builder, Location loc, Operation *inOp,
    const QASMVectorToQuakeVector &QASMToVectors) {
  // Value qubits, IDQASMMLIR mlirQubits) {
  bool isAdj = false;
  std::vector<Value> parameters = {};
  std::vector<Value> controls = {};
  std::vector<Value> targets = {};
  // Defining the builder
  builder.setInsertionPoint(inOp); // Set insertion before return
  // Print the gate type (identifier)
#ifdef DEBUG
  std::cout << "Gate Type: " << gateCall->identifier << std::endl;
  std::cout << "Arguments size: " << gateCall->arguments.size() << std::endl;
#endif
  // Print parameters (arguments)
  if (!gateCall->arguments.empty()) {
#ifdef DEBUG
    std::cout << "Parameters: ";
#endif
    for (const auto &arg : gateCall->arguments) {
      double argVal = evaluateExpression(arg);
      Value argMlirVal = createFloatValue(builder, loc, argVal);
      parameters.push_back(argMlirVal);
#ifdef DEBUG
      std::cout << argVal << " ";
#endif
    }
#ifdef DEBUG
    std::cout << std::endl;
#endif
  }
  // Print operands and their types (control or target)
  if (!gateCall->operands.empty()) {
#ifdef DEBUG
    std::cout << "Operands: " << std::endl;
#endif
    // Determine the number of controls
    size_t numControls = 0;
    for (const auto &modifier : gateCall->modifiers) {
      if (auto ctrlMod =
              std::dynamic_pointer_cast<qasm3::CtrlGateModifier>(modifier)) {
        if (ctrlMod->expression) {
          if (auto constantExpr = std::dynamic_pointer_cast<qasm3::Constant>(
                  ctrlMod->expression)) {
            numControls = constantExpr->getSInt();
#ifdef DEBUG
            std::cout << "numControls " << numControls << "\n";
#endif
            break;
          }
        }
      }
    }
    // If no explicit controls, check if it's a multi-qubit gate with implicit
    // controls
    if (numControls == 0 && isMultiQubitGate(gateCall->identifier)) {
      numControls = getNumControls(gateCall->identifier);
    }
    // Iterate over operands and classify them as controls or targets
    for (size_t i = 0; i < gateCall->operands.size(); ++i) {
      const auto &operand = gateCall->operands[i];
#ifdef DEBUG
      std::cout << "  - " << operand->identifier;
#endif
      // get the qubit index
      int qubitOp = -1;
      if (auto constantExprOp =
              std::dynamic_pointer_cast<qasm3::Constant>(operand->expression))
        qubitOp = constantExprOp->getSInt();
      assert(qubitOp != -1 && "Fatal error, this must not happen!");
      Value selectedVector = QASMToVectors.at(std::string(operand->identifier));
      int selectedQubit = qubitOp;
#ifdef DEBUG
      if (operand->expression) {
        std::cout << "[" << qubitOp << "]";
      }
#endif
      if (i < numControls) {
        auto controlQubit = builder.create<quake::ExtractRefOp>(
            loc, selectedVector, selectedQubit);
        controls.push_back(controlQubit);
#ifdef DEBUG
        std::cout << " (Control)";
#endif
      } else {
        auto targetQubit = builder.create<quake::ExtractRefOp>(
            loc, selectedVector, selectedQubit);
        targets.push_back(targetQubit);
#ifdef DEBUG
        std::cout << " (Target)";
#endif
      }
#ifdef DEBUG
      std::cout << std::endl;
#endif
    }
  }
  if (std::regex pattern("dg");
      std::regex_search(std::string(gateCall->identifier), pattern)) {
    isAdj = true;
  }
  std::string gateId = gateCall->identifier;
  std::ranges::transform(gateId, gateId.begin(),
                         [](unsigned char c) { return std::tolower(c); });
  insertQASMGateIntoQuakeModule(gateId, builder, loc, parameters, controls,
                                targets, isAdj);
#ifdef DEBUG
  std::cout << "-------------------------" << std::endl;
#endif
}

// Function that parses a given AST/QASM and inserts measurements into a
// MLIR/Quake
void mqss::interfaces::parseAndInsertMeasurements(
    const std::vector<std::shared_ptr<qasm3::Statement>> &statements,
    OpBuilder &builder, Location loc, Operation *inOp,
    const QASMVectorToQuakeVector &QASMToVectors) {
  std::map<int, std::pair<std::string, int>> ClassicalRegToQVector = {};
  // Value allocatedQubits, IDQASMMLIR mlirQubits) {
  // Defining the builder
  builder.setInsertionPoint(inOp); // Set insertion before return
  // llvm::outs() << "Printing measurements!\n";
  for (const auto &statement : statements) {
    // llvm::outs() << "Statement Type: " << typeid(*statement).name() << "\n";
    //   Check if the statement is a MeasureStatement
    if (auto assignmentStmt =
            std::dynamic_pointer_cast<qasm3::AssignmentStatement>(statement)) {
      // llvm::outs() << "Found AssignmentStatement\n";
      std::string classicalRegister;
      size_t classicalIndex = 0;
      if (assignmentStmt->identifier)
        classicalRegister = assignmentStmt->identifier->getName();
      if (auto idxExpr = std::dynamic_pointer_cast<qasm3::Constant>(
              assignmentStmt->indexExpression)) {
        classicalIndex = idxExpr->getSInt();
      } else {
        llvm::errs() << "Error: indexExpression is not a Constant.\n";
        assert(false && "Unsupported classical index expression");
      }
#ifdef DEBUG
      llvm::outs() << "Measurement result goes to: " << classicalRegister << "["
                   << classicalIndex << "]\n";
#endif
      if (assignmentStmt->expression) {
        // llvm::outs() << "Expression Type: " <<
        // typeid(assignmentStmt->expression).name() << "\n";
        //  Check if it's a DeclarationExpression (which holds the initializer)
        if (auto declExpr =
                std::dynamic_pointer_cast<qasm3::DeclarationExpression>(
                    assignmentStmt->expression)) {
          // llvm::outs() << "Found DeclarationExpression\n";
          //  Check if the initializer is a MeasureExpression
          if (auto measureExpr =
                  std::dynamic_pointer_cast<qasm3::MeasureExpression>(
                      declExpr->expression)) {
            if (measureExpr->gate) {
              std::string qVector = measureExpr->gate->identifier;
              // llvm::outs() << "Measured Qubit: " << qubit << "\n";
              if (measureExpr->gate->expression) {
                // llvm::outs() << "Has expression\n";
                if (auto operand = std::dynamic_pointer_cast<qasm3::Constant>(
                        measureExpr->gate->expression)) {
                  size_t localQubit = operand->getSInt();
                  ClassicalRegToQVector.emplace(
                      classicalIndex, std::make_pair(qVector, localQubit));
                  // llvm::outs() << "Operand Identifier: " <<
                  // operand->getSInt() << "\n";
                }
              }
            } else
              assert(false && "Measurement has not qubit associated to it!");
          }
        }
      }
    }
  }
  for (const auto &qVectorPair : ClassicalRegToQVector | std::views::values) {
    auto [qVector, qubit] = qVectorPair;
    // using QASMVectorToQuakeVector = std::unordered_map<std::string, Value>;
    Value selectedQuakeVector = QASMToVectors.at(qVector);

    // insert measurement
    auto measRef =
        builder.create<quake::ExtractRefOp>(loc, selectedQuakeVector, qubit);
    SmallVector<Value> targetValues = {measRef};
    Type measTy = quake::MeasureType::get(builder.getContext());
    builder.create<quake::MzOp>(loc, measTy, targetValues).getMeasOut();
  }
}