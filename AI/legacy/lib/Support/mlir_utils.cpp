#include "Support/mlir_utils.hpp"

// Other includes
#include "Interfaces/Constants.hpp"
#include "common/RuntimeMLIR.h"

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

// Cudaq includes
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

// Stdandard library includes
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>

using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::arith::ConstantOp;

namespace mqss::support::quakeDialect {
std::vector<double> params_to_angles(std::vector<double> params) {
  for (int i = 0; i < params.size(); i++) {
    double angle = std::fmod(params[i] + PI, TWO_PI);
    if (angle < 0) {
      angle += TWO_PI;
    }
    params[i] = angle - PI;
  }
  return params;
}

std::string getOperationName(Operation *op) {
  if (!op) {
    return "nullptr";
  }
  return op->getName().getIdentifier().getValue().str();
}

std::string getOnlyGateName(Operation *op) {
  if (!isGate(op)) {
    return "";
  }
  auto [_, gateName] = op->getName().getStringRef().split('.');
  return std::string(gateName);
}

std::tuple<ModuleOp, MLIRContext *>
extractMLIRContext(const std::string &quakeModule) {
  auto contextPtr = cudaq::initializeMLIR();
  MLIRContext &context = *contextPtr.get();

  // Get the quake representation of the kernel
  auto quakeCode = quakeModule;
  auto m_module = mlir::parseSourceString<ModuleOp>(quakeCode, &context);
  if (!m_module) {
    throw std::runtime_error("Module cannot be parsed in extractMLIRContext.");
  }
  return std::make_tuple(m_module.release(), contextPtr.release());
}

std::pair<ModuleOp, std::unique_ptr<MLIRContext *>>
extractModuleOpAndContextPointer(const std::string &quakeModule) {
  auto contextPtr = cudaq::initializeMLIR();
  MLIRContext &context = *contextPtr.get();
  // Get the quake representation of the kernel
  auto quakeCode = quakeModule;
  auto m_module = mlir::parseSourceString<ModuleOp>(quakeCode, &context);
  if (!m_module) {
    throw std::runtime_error(
        "Module cannot be parsed in extractModuleOpAndContextPointer.");
  }
  return {m_module.release(),
          std::make_unique<MLIRContext *>(contextPtr.release())};
}

std::string readFileToString(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return "";
  }
  std::ostringstream fileContents;
  fileContents << file.rdbuf();
  return fileContents.str();
}

std::vector<int> getMeasurementTargets(Operation *op, int nr_qubits) {
  if (!isMeasurement(op)) {
    return {};
  }
  std::vector<int> targets = {};
  if (op->getOpOperands().size() != 1) {
    throw std::runtime_error("Measurement gate op is ambiguous.");
  }
  if (auto operand = op->getOpOperands().front().get();
      operand.getType().isa<quake::RefType>()) {
    auto targetIndexOpt =
        extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
    if (!targetIndexOpt.has_value()) {
      return {};
    }
    targets.push_back(targetIndexOpt.value());
  } else if (operand.getType().isa<quake::VeqType>()) {
    // Because this function only works for a single allocation, the
    // reference to a Veq will reference all allocated qubits in the
    // range [0, nr_qubits-1].
    for (int qubitIndex = 0; qubitIndex < nr_qubits; qubitIndex++) {
      targets.push_back(qubitIndex);
    }
  } else {
    throw std::runtime_error("Measurement gate op has unsupported operand.");
  }
  return targets;
}

std::tuple<unsigned int, unsigned int, unsigned int>
getQubitsInstructionsDepth(FuncOp circuit) {
  unsigned int nr_qubits = 0;
  unsigned int nr_gates = 0;
  std::vector<unsigned int> depths;

  circuit.walk([&](Operation *op) {
    if (isa<quake::AllocaOp>(op)) {
      if (auto allocOp = dyn_cast<quake::AllocaOp>(op);
          allocOp.getType().dyn_cast<quake::RefType>()) {
        nr_qubits += 1;
      } else if (auto qvecType = allocOp.getType().dyn_cast<quake::VeqType>()) {
        nr_qubits += qvecType.getSize();
      }
      depths.resize(nr_qubits, 0);
      return;
    }
    if (!isGate(op)) {
      return;
    }
    if (isMeasurement(op)) {
      for (Value operand : op->getOperands()) {
        if (operand.getType().isa<quake::RefType>()) {
          auto qubitIndexOpt =
              extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
          if (!qubitIndexOpt.has_value()) {
            continue;
          }
          if (int qubitIndex = qubitIndexOpt.value();
              0 <= qubitIndex && qubitIndex < nr_qubits) {
            nr_gates++;
            depths[qubitIndex]++;
          }
        } else if (operand.getType().isa<quake::VeqType>()) {
          // Because this function only works for a single allocation, the
          // reference to a Veq will reference all allocated qubits in the
          // range [0, nrQubits-1].
          for (int qubitIndex = 0; qubitIndex < nr_qubits; qubitIndex++) {
            depths[qubitIndex]++;
          }
          nr_gates += operand.getType().dyn_cast<quake::VeqType>().getSize();
        }
      }
      return;
    }
    nr_gates++;
    auto gate = dyn_cast<quake::OperatorInterface>(op);
    std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
    std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
    targets.insert(targets.end(), controls.begin(), controls.end());
    unsigned int max_depth = 0;
    for (int qubit : targets) {
      max_depth = std::max(max_depth, depths[qubit]);
    }
    for (int qubit : targets) {
      depths[qubit] = max_depth + 1;
    }
  });
  if (depths.empty()) {
    // If a malformed kernel skips allocation entirely, we assume zero depth.
    return {nr_qubits, nr_gates, 0};
  }
  return {nr_qubits, nr_gates, *std::ranges::max_element(depths)};
}

std::string vectorToString(const std::vector<int> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

std::string vectorToString(const std::vector<double> &vec) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

std::string setToString(const std::unordered_set<std::string> &set) {
  std::ostringstream oss;
  bool first = true;
  for (const auto &element : set) {
    if (!first)
      oss << ", ";
    oss << element;
    first = false;
  }
  return oss.str();
}

std::string valueRangeToString(ValueRange range) {
  std::string out;
  llvm::raw_string_ostream rso(out);
  rso << "[";
  for (size_t i = 0; i < range.size(); ++i) {
    rso << range[i];
    if (i + 1 < range.size())
      rso << ", ";
  }
  rso << "]";
  return rso.str();
}

bool isMeasurement(Operation *op) {
  if (op->getDialect()->getNamespace() != "quake") {
    return false;
  }
  return isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op);
}

bool isOperation(Operation *op) {
  if (op->getDialect()->getNamespace() != "quake") {
    return false;
  }
  return isa<quake::XOp>(op) || isa<quake::YOp>(op) || isa<quake::ZOp>(op) ||
         isa<quake::HOp>(op) || isa<quake::SOp>(op) || isa<quake::TOp>(op) ||
         isa<quake::RxOp>(op) || isa<quake::RyOp>(op) || isa<quake::RzOp>(op) ||
         isa<quake::SwapOp>(op) || isa<quake::R1Op>(op) ||
         isa<quake::U2Op>(op) || isa<quake::U3Op>(op) ||
         isa<quake::PhasedRxOp>(op);
}

bool isGate(Operation *op) {
  if (op->getDialect()->getNamespace() != "quake") {
    return false;
  }
  return isMeasurement(op) || isOperation(op);
}

int getNumberOfAllocations(FuncOp circuit) {
  int nrAllocations = 0;
  circuit.walk([&](Operation *op) {
    // An allocation is allowed to be a single-qubit or multi-qubit (Veq)
    // allocation. However, each specific allocation, even the veq allocation
    // is just one allocation operation.
    if (isa<quake::AllocaOp>(op)) {
      nrAllocations++;
    }
  });
  return nrAllocations;
}

std::vector<double> getOperationParameters(Operation *op) {
  if (!isGate(op)) {
    return {};
  }
  auto gate = dyn_cast<quake::OperatorInterface>(op);
  std::vector<double> parameters;
  for (auto parameter : gate.getParameters()) {
    std::optional<double> param =
        extractDoubleArgumentValue(parameter.getDefiningOp());
    if (param.has_value()) {
      parameters.push_back(param.value());
    }
  }
  return parameters;
}

// Given a OpBuilder and a double value, it inserts a double in the mlir
// module pointer by the OpBuilder and returns the inserted Value
Value createFloatValue(OpBuilder &builder, const Location loc,
                       const double value) {
  // Create a constant value (20.0 of type f64)
  auto valueAttr = builder.getFloatAttr(builder.getF64Type(), value);
  auto constantOp = builder.create<ConstantOp>(loc, valueAttr);
  return constantOp.getResult();
}

std::optional<double> extractDoubleArgumentValue(Operation *op) {
  if (auto constantOp = dyn_cast<ConstantOp>(op)) {
    if (auto floatAttr = constantOp.getValue().dyn_cast<FloatAttr>()) {
      return floatAttr.getValueAsDouble();
    }
  }
  return std::nullopt;
}

// TODO: return -1 is not good idea
// Given an ExtractRefOp, it extracts the integer of the index pointing that
// reference (qubit index), returns -1 when fail
std::optional<int64_t> extractIndexFromQuakeExtractRefOp(Operation *op) {
  if (auto extractRefOp = llvm::dyn_cast<quake::ExtractRefOp>(op)) {
    auto rawIndexAttr = extractRefOp->getAttrOfType<IntegerAttr>("rawIndex");
    return rawIndexAttr.getInt();
  }
  return std::nullopt;
}

// function to get the number of qubits in a given quantum kernel
unsigned int getNumberOfQubits(FuncOp circuit) {
  unsigned int numQubits = 0;
  circuit.walk([&](quake::AllocaOp allocOp) {
    if (allocOp.getType().dyn_cast<quake::RefType>()) {
      numQubits += 1;
    } else if (auto qvecType = allocOp.getType().dyn_cast<quake::VeqType>()) {
      numQubits += qvecType.getSize();
    }
  });
  return numQubits;
}

unsigned int getNumberOfGates(FuncOp circuit) {
  unsigned int nrQubits = getNumberOfQubits(circuit);
  if (nrQubits == 0) {
    return 0;
  }
  int nrGates = 0;
  circuit.walk([&](Operation *op) {
    if (!isGate(op)) {
      return;
    }
    if (isMeasurement(op)) {
      for (auto operand : op->getOperands()) {
        if (operand.getType().isa<quake::RefType>()) {
          auto qubitIndexOpt =
              extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
          if (!qubitIndexOpt.has_value()) {
            continue;
          }
          if (int qubitIndex = qubitIndexOpt.value();
              0 <= qubitIndex && qubitIndex < nrQubits) {
            nrGates++;
          }
        } else if (operand.getType().isa<quake::VeqType>()) {
          nrGates += operand.getType().dyn_cast<quake::VeqType>().getSize();
        }
      }
    } else {
      nrGates++;
    }
  });
  return nrGates;
}

unsigned int getCircuitDepth(FuncOp circuit) {
  unsigned int nrQubits = getNumberOfQubits(circuit);
  if (nrQubits == 0) {
    return 0;
  }
  if (getNumberOfAllocations(circuit) != 1) {
    std::cerr
        << "Function getCircuitDepth not implemented for multiple allocations"
        << std::endl;
    return -1;
  }

  std::vector depths(nrQubits, 0u);
  circuit.walk([&](Operation *op) {
    if (!isGate(op)) {
      return;
    }
    if (isMeasurement(op)) {
      for (auto operand : op->getOperands()) {
        if (operand.getType().isa<quake::RefType>()) {
          auto qubitIndexOpt =
              extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
          if (!qubitIndexOpt.has_value()) {
            continue;
          }
          if (int qubitIndex = qubitIndexOpt.value();
              0 <= qubitIndex && qubitIndex < nrQubits) {
            depths[qubitIndex]++;
          }
        } else if (operand.getType().isa<quake::VeqType>()) {
          // Because this function only works for a single allocation, the
          // reference to a Veq will reference all allocated qubits in the
          // range [0, nrQubits-1].
          for (int qubitIndex = 0; qubitIndex < nrQubits; qubitIndex++) {
            depths[qubitIndex]++;
          }
        }
      }
    } else {
      auto gate = dyn_cast<quake::OperatorInterface>(op);
      std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
      std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
      targets.insert(targets.end(), controls.begin(), controls.end());
      unsigned int max_depth = 0;
      for (int qubit : targets) {
        max_depth = std::max(max_depth, depths[qubit]);
      }
      for (int qubit : targets) {
        depths[qubit] = max_depth + 1;
      }
    }
  });
  return *std::ranges::max_element(depths);
}

// Function to get the number of classical bits allocated in a given
// quantum kernel, it also stores information of the qubit position
int getNumberOfClassicalBits(FuncOp circuit, std::map<int, int> &measurements) {
  if (getNumberOfAllocations(circuit) != 1) {
    std::cerr << "Function getNumberOfClassicalBits not implemented for "
                 "multiple allocations"
              << std::endl;
    return -1;
  }
  int numBits = 0;
  circuit.walk([&](Operation *op) {
    if (isMeasurement(op)) {
      for (auto operand : op->getOperands()) {
        // Check if it's qubit reference
        if (operand.getType().isa<quake::RefType>()) {
          auto qubitIndexOpt =
              extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
          if (!qubitIndexOpt.has_value()) {
            continue;
          }
          int qubitIndex = qubitIndexOpt.value();
          assert(qubitIndex != -1 && "Non valid qubit index for measurement!");
          measurements[qubitIndex] = numBits;
          numBits += 1;
        } else if (operand.getType().isa<quake::VeqType>()) {
          auto qvecType = operand.getType().dyn_cast<quake::VeqType>();
          int start = numBits;
          numBits += qvecType.getSize();
          // Assume Veq qubits are sequential indices matching global bit
          // positions; may not hold for sliced/concatenated Veq
          // TODO: compute actual global indices.
          for (int i = start; i < numBits; i++) {
            measurements[i] = i;
          }
        }
      }
    }
  });
  return numBits;
}

// Function to get the number of classical bits allocated in
// a given quantum kernel
int getNumberOfClassicalBits(FuncOp circuit) {
  if (getNumberOfAllocations(circuit) != 1) {
    std::cerr << "Function getNumberOfClassicalBits not implemented for "
                 "multiple allocations"
              << std::endl;
    return -1;
  }
  int numBits = 0;
  circuit.walk([&](Operation *op) {
    if (isa<quake::MxOp>(op) || isa<quake::MyOp>(op) || isa<quake::MzOp>(op)) {
      for (auto operand : op->getOperands()) {
        if (operand.getType().isa<quake::RefType>()) {
          // Check if it's a qubit reference
          auto qubitIndexOpt =
              extractIndexFromQuakeExtractRefOp(operand.getDefiningOp());
          if (!qubitIndexOpt.has_value()) {
            continue;
          }
          int qubitIndex = qubitIndexOpt.value();
          assert(qubitIndex != -1 && "Non valid qubit index for measurement!");
          numBits += 1;
        } else if (operand.getType().isa<quake::VeqType>()) {
          auto qvecType = operand.getType().dyn_cast<quake::VeqType>();
          numBits += qvecType.getSize();
        }
      }
    }
  });
  return numBits;
}

// Function that get the indices of the Value objectes in array
std::vector<int> getIndicesOfValueRange(const ValueRange array) {
  std::vector<int> indices;
  for (auto value : array) {
    auto qubitIndexOpt =
        extractIndexFromQuakeExtractRefOp(value.getDefiningOp());
    if (!qubitIndexOpt.has_value()) {
      continue;
    }
    int qubitIndex = qubitIndexOpt.value();
    indices.push_back(qubitIndex);
  }
  return indices;
}

// At the moment, it is assumed that the parameters are of type Double
std::vector<double> getParametersValues(const ValueRange array) {
  std::vector<double> parameters;
  for (auto value : array) {
    auto paramOpt = extractDoubleArgumentValue(value.getDefiningOp());
    if (!paramOpt.has_value()) {
      continue;
    }
    double param = paramOpt.value();
    parameters.push_back(param);
  }
  return parameters;
}

// Get the previous operation on a given TargeQubit, starting from
// currentOp
Operation *getPreviousOperationOnTarget(Operation *currentOp,
                                        Value targetQubit) {
  auto targetQCurrOpt =
      extractIndexFromQuakeExtractRefOp(targetQubit.getDefiningOp());
  if (!targetQCurrOpt.has_value()) {
    return nullptr;
  }
  int targetQCurr = targetQCurrOpt.value();
  // Start from the previous operation
  Operation *prevOp = currentOp->getPrevNode();
  // Iterate through the previous operations in the block
  while (prevOp) {
    // Check if the operation has a target qubit and matches the given target
    if (auto quakeOp = dyn_cast<quake::OperatorInterface>(prevOp)) {
      for (Value target : quakeOp.getTargets()) {
        auto targetQPrevOpt =
            extractIndexFromQuakeExtractRefOp(target.getDefiningOp());
        if (!targetQPrevOpt.has_value()) {
          continue;
        }
        if (int targetQPrev = targetQPrevOpt.value();
            targetQCurr == targetQPrev)
          return prevOp;
      }
      for (Value control : quakeOp.getControls()) {
        auto controlQPrevOpt =
            extractIndexFromQuakeExtractRefOp(control.getDefiningOp());
        if (!controlQPrevOpt.has_value()) {
          continue;
        }
        if (int controlQPrev = controlQPrevOpt.value();
            targetQCurr == controlQPrev)
          return prevOp;
      }
    }
    // Move to the previous operation
    prevOp = prevOp->getPrevNode();
  }
  return nullptr; // No matching previous operation found
}

Operation *getNextOperationOnTarget(Operation *currentOp, Value targetQubit) {
  auto targetQCurrOpt =
      extractIndexFromQuakeExtractRefOp(targetQubit.getDefiningOp());
  if (!targetQCurrOpt.has_value()) {
    return nullptr;
  }
  int targetQCurr = targetQCurrOpt.value();
  Operation *nextOp = currentOp->getNextNode();

  while (nextOp) {
    if (auto quakeOp = dyn_cast<quake::OperatorInterface>(nextOp)) {
      for (Value target : quakeOp.getTargets()) {
        auto targetQNextOpt =
            extractIndexFromQuakeExtractRefOp(target.getDefiningOp());
        if (!targetQNextOpt.has_value()) {
          continue;
        }
        if (int targetQNext = targetQNextOpt.value();
            targetQCurr == targetQNext) {
          return nextOp;
        }
      }
      for (Value control : quakeOp.getControls()) {
        auto controlQNextOpt =
            extractIndexFromQuakeExtractRefOp(control.getDefiningOp());
        if (!controlQNextOpt.has_value()) {
          continue;
        }
        if (int controlQNext = controlQNextOpt.value();
            targetQCurr == controlQNext) {
          return nextOp;
        }
      }
    }

    nextOp = nextOp->getNextNode();
  }
  return nullptr;
}

} // namespace mqss::support::quakeDialect