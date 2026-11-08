#include "Interfaces/QuakeToLinAlg.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Casting.h"

#include <complex>
#include <iostream>
#include <regex>

using namespace mlir;
using namespace complex;
using namespace utils;
using namespace mqss::support::quakeDialect;

bool hasFunc(ModuleOp module, StringRef name) {
  return static_cast<bool>(module.lookupSymbol<FuncOp>(name));
}

FuncOp getFuncOp(ModuleOp module, StringRef name) {
  return module.lookupSymbol<FuncOp>(name);
}

/// Two‑operand multiply helper.
/// ops[0] and ops[1] must be the same element type (complex<f32|f64>)
/// and ranks among {0,1,2}. Returns a Value of the multiplied result.
Value insertMul(OpBuilder &builder, Location loc, ArrayRef<Value> ops) {
  assert(ops.size() == 2 && "insertMul requires exactly two operands");

  // Fetch the tensor type for operand 0 (we assume both have same shape/type).
  auto rt0 = ops[0].getType().dyn_cast<RankedTensorType>();
  auto rt1 = ops[1].getType().dyn_cast<RankedTensorType>();
  assert(rt0 && rt1 && "Operands must be RankedTensorType");

  auto rank0 = rt0.getRank();
  auto rank1 = rt1.getRank();
  auto eltType = rt0.getElementType().dyn_cast<ComplexType>();
  assert(eltType && eltType == rt1.getElementType().dyn_cast<ComplexType>() &&
      "Element types must match and be complex");

  // --- Case A: Scalar (rank0 == 0 && rank1 == 0) ---
  if (rank0 == 0 && rank1 == 0) {
    return builder.create<MulOp>(loc, eltType, ops[0], ops[1])
        .getResult();
  }

  // --- Case B: Vector (rank0 == 1 && rank1 == 1, same shape) ---
  if (rank0 == 1 && rank1 == 1 && rt0.getShape() == rt1.getShape()) {
    // Create empty output tensor
    Value resultTensor =
        builder.create<tensor::EmptyOp>(loc, rt0.getShape(), eltType);

    auto ctx = builder.getContext();
    auto dimExpr = getAffineDimExpr(0, ctx);
    auto map =
        AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {dimExpr});
    SmallVector indexingMaps = {map, map, map};
    SmallVector iterTypes = {IteratorType::parallel};

    auto genericOp = builder.create<linalg::GenericOp>(
        loc, TypeRange{rt0}, ValueRange{ops[0], ops[1]},
        ValueRange{resultTensor}, indexingMaps, iterTypes,
        /*doc=*/"",
        /*libraryCall=*/"",
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value prod = nestedBuilder.create<MulOp>(
              nestedLoc, eltType, args[0], args[1]);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, prod);
        });

    return genericOp.getResult(0);
  }

  // --- Case C: Matrix × Vector → Matvec (rank0==2, rank1==1) ---
  if (rank0 == 2 && rank1 == 1 && rt0.getDimSize(1) == rt1.getDimSize(0)) {
    // Init result tensor< M x complex >
    int64_t M = rt0.getDimSize(0);
    auto resultType = RankedTensorType::get({M}, eltType);
    Value init = builder.create<tensor::EmptyOp>(
        loc, ArrayRef{M}, eltType);
    return builder
        .create<linalg::MatvecOp>(loc, resultType, ValueRange{ops[0], ops[1]},
                                  ValueRange{init})
        .getResult(0);
  }

  // --- Case D: Matrix × Matrix → Matmul (rank0==2, rank1==2) ---
  if (rank0 == 2 && rank1 == 2 && rt0.getDimSize(1) == rt1.getDimSize(0)) {
    int64_t M = rt0.getDimSize(0), N = rt1.getDimSize(1);
    auto resultType = RankedTensorType::get({M, N}, eltType);
    Value init = builder.create<tensor::EmptyOp>(
        loc, ArrayRef{M, N}, eltType);
    return builder.create<linalg::MatmulOp>(
        loc, resultType, ValueRange{ops[0], ops[1]},
        ValueRange{init}).getResult(0);
  }

  llvm_unreachable("Unsupported operand shapes for insertMul");
}

/// Insert an N‑ary multiply by folding the 2‑operand case.
/// Requires at least 2 operands.
Value insertMulN(OpBuilder &builder, Location loc, ArrayRef<Value> ops) {
  assert(ops.size() >= 2 && "Need at least two operands to multiply");

  // Start by multiplying the first two:
  SmallVector<Value, 4> two = {ops[0], ops[1]};
  Value acc = insertMul(builder, loc, two);

  // Then fold in the rest
  for (size_t i = 2, e = ops.size(); i < e; ++i) {
    two[0] = acc;
    two[1] = ops[i];
    acc = insertMul(builder, loc, two);
  }

  return acc;
}

Value initializeQubits(
    FuncOp gpuFunction, OpBuilder &builder, int numberOfQubits,
    RankedTensorType tensorType) {
  Block &entryBlock = gpuFunction.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);

  std::vector<std::complex<double> > data;
  for (int i = 0; i < std::pow(2, numberOfQubits); i++)
    data.push_back({0.0, 0.0});

  DenseElementsAttr initAttr =
      DenseElementsAttr::get(tensorType, ArrayRef(data));
  Location loc = builder.getUnknownLoc();

  auto constantOp =
      builder.create<arith::ConstantOp>(loc, tensorType, initAttr);
  return constantOp;
}

FuncOp inlineFunction(
    ModuleOp &module, std::string operationName, RankedTensorType matrixType,
    const long numberParameters) {
  // if the function is already in the module,
  // then return it
  if (hasFunc(module, operationName))
    return getFuncOp(module, operationName);
  OpBuilder builder(module.getBodyRegion());
  Location loc = builder.getUnknownLoc();
  // if the gate has parameters the signature function has to
  // accept a vector of size of the parameters
  FunctionType funcType;
  if (numberParameters > 0) {
    auto elementType = builder.getF64Type();
    RankedTensorType tensorType =
        RankedTensorType::get({numberParameters}, elementType);
    funcType = builder.getFunctionType({tensorType}, {matrixType});
  } else {
    funcType = builder.getFunctionType({}, {matrixType});
  }

  auto operationFunction =
      builder.create<FuncOp>(loc, operationName, funcType);
  operationFunction.setPrivate(); // Optional: make it private visibility
  operationFunction.setVisibility(SymbolTable::Visibility::Private);
  // Erase the body to make it external (opaque)
  operationFunction.eraseBody();
  return operationFunction;
}

Value createDoubleTensor(OpBuilder &builder, Location loc,
                         const std::vector<double> &dataVec) {
  // Create the tensor type: tensor<Nxf64>
  auto elementType = builder.getF64Type();
  auto tensorType = RankedTensorType::get(
      {static_cast<int64_t>(dataVec.size())}, elementType);

  // Convert to APFloat and build a DenseElementsAttr
  SmallVector<APFloat, 4> apValues;
  for (double val : dataVec)
    apValues.emplace_back(APFloat(val));

  auto attr = DenseElementsAttr::get(tensorType, apValues);

  // Create the constant operation
  return builder.create<arith::ConstantOp>(loc, tensorType, attr)
      .getResult();
}

Value mqss::interfaces::convertQuakeToLinAlg(
    ModuleOp module, FuncOp quakeFunction, OpBuilder &builder,
    FuncOp gpuFunction, RankedTensorType tensorType,
    RankedTensorType matrixType, int numberOfQubits) {
  // insert the initial state of the circuit
  Value state =
      initializeQubits(gpuFunction, builder, numberOfQubits, tensorType);
  // iterate the function to see if it is quake
  quakeFunction.walk([&](Operation *op) {
    auto gate = dyn_cast<quake::OperatorInterface>(op);
    if (!gate)
      return;
    // then, the operation is a quake gate
    StringRef opName = op->getName().getStringRef();
    std::regex pattern("^quake\\.");
    std::string gateName = std::regex_replace(opName.str(), pattern, "");
    // get the target and get the controls
    std::vector<int> controls = getIndicesOfValueRange(gate.getControls());
    std::vector<int> targets = getIndicesOfValueRange(gate.getTargets());
    std::vector<double> params = getParametersValues(gate.getParameters());
    std::string controlString = "[";
    std::string targetString = "[";
    for (int i = 0; i < controls.size(); i++)
      controlString += std::to_string(controls[i]);
    controlString += "]";
    for (int i = 0; i < targets.size(); i++)
      targetString += std::to_string(targets[i]);
    targetString += "]";
    std::string functionName = gateName + "_" + std::to_string(numberOfQubits) +
                               "qubits_" + "control" + controlString +
                               "_target" + targetString;
    // inline the function
    Location loc = builder.getUnknownLoc();
    // Emit func.call to the private function
    Value matrixValue;
    if (params.size() > 0) {
      Value arguments = createDoubleTensor(builder, loc, params);
      matrixValue = builder.create<func::CallOp>(
          loc, functionName, matrixType, arguments).getResult(0);
    } else {
      matrixValue = builder.create<func::CallOp>(
          loc, functionName, matrixType, ValueRange{}).getResult(0);
    }
    SmallVector<Value> operands;
    operands.push_back(matrixValue);
    operands.push_back(state);
    state = insertMulN(builder, loc, operands);
  });
  return state;
}