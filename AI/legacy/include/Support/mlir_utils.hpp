#ifndef MLIR_UTILS_HPP
#define MLIR_UTILS_HPP

// MLIR includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>

// Stdandard library includes
#include <string>
#include <tuple>
#include <unordered_set>

////////////////////////////////////////////////////////////////////////////////
/// Libtorch c10::ArrayRef conflicts with llvm::ArrayRef included in the mlir
/// namespace, so every mlir type has to be included seperately.
using mlir::Location;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::Value;
using mlir::ValueRange;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
////////////////////////////////////////////////////////////////////////////////

namespace mqss::support::quakeDialect {
constexpr double TWO_PI = 6.28318530717958647692;
constexpr float TWO_PI_FLOAT = 6.28318530717958647692f;

/**
 *
 * @param params
 * @return
 */
std::vector<double> params_to_angles(std::vector<double> params);

/**
 * @param op The operation to extract the name from.
 * @return The name of the operations as a std::string.
 */
std::string getOperationName(Operation *op);

/**
 *
 * @param op
 * @return
 */
std::string getOnlyGateName(Operation *op);

/**
 *
 * @param op
 * @return
 */
bool isMeasurement(Operation *op);

/**
 *
 * @param op
 * @return
 */
bool isOperation(Operation *op);

/**
 *
 * @param op
 * @return
 */
bool isGate(Operation *op);

/**
 * Extracts a MLIR module operation and the MLIR context from the quake module
 * which is the string content of a quake file.
 * @param quakeModule The string contents of the quake file.
 * @return The MLIR module and context.
 */
std::tuple<ModuleOp, MLIRContext *>
extractMLIRContext(const std::string &quakeModule);

/**
 *
 * @param quakeModule
 * @return
 */
std::pair<ModuleOp, std::unique_ptr<MLIRContext *>>
extractModuleOpAndContextPointer(const std::string &quakeModule);

/**
 * Extracts the string contents of a quake file.
 * @param filename The name of the quake file.
 * @return The string contents of the quake file.
 */
std::string readFileToString(const std::string &filename);

/**
 *
 * @param op
 * @param nr_qubits
 * @return
 */
std::vector<int> getMeasurementTargets(Operation *op, int nr_qubits);

/**
 *
 * @param op
 * @param nr_qubits
 * @return
 */
std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>
getNoneMeasurementControlsTargetsParams(Operation *op, int nr_qubits);

/**
 *
 * @param circuit
 * @return
 */
std::tuple<unsigned int, unsigned int, unsigned int>
getQubitsInstructionsDepth(FuncOp circuit);

/**
 *
 * @param vec
 * @return
 */
std::string vectorToString(const std::vector<int> &vec);

/**
 *
 * @param vec
 * @return
 */
std::string vectorToString(const std::vector<double> &vec);

/**
 *
 * @param range
 * @return
 */
std::string valueRangeToString(ValueRange range);

/**
 *
 * @param set
 * @return
 */
std::string setToString(const std::unordered_set<std::string> &set);

/**
 *
 * @param op
 * @return
 */
std::vector<double> getOperationParameters(Operation *op);

/**
  @brief Function that creates an `Value` associated to a numeric value.
  @details This functions appends an `Value` into an MLIR module
 associated to the input `OpBuilder`.
  @param[out] builder is an `OpBuilder` object associated with a MLIR module.
 It is used to insert new instructions to the corresponding MLIR module.
  @param[in] loc is the location of the new inserted instruction.
  @param[in] value is the numeric value to be defined into the MLIR module.
  @return an `Value` object of the inserted numerical value.
*/
Value createFloatValue(OpBuilder &builder, Location loc, double value);

/**
  @brief Function that extracts a double numeric value from a numeric value in
  an MLIR module.
  @details This functions extracts a `double` from an MLIR `Operation`.
  @param[in] op is the MLIR `Operation` containing a numerical value.
  @return a `double` with the numerical value of op.
*/
std::optional<double> extractDoubleArgumentValue(Operation *op);

/**
  @brief Function that extracts an index of a given `ExtractRefOp` operation.
  @details Given an `ExtractRefOp`, this function extracts the integer of the
  index pointing that reference (qubit index), returns -1 when fail.
  @param[in] op is the MLIR `ExtractRefOp`.
  @return a `int` with the index of the given `ExtractRefOp`.
*/
std::optional<int64_t> extractIndexFromQuakeExtractRefOp(Operation *op);

/**
  @brief Function that get the number of qubits used by a given quantum kernel.
  @details Given a `FuncOp` that stores a quantum kernel in Quake. This
  function returns the number of declared qubits within the given quantum
  kernel.
  @param[in] circuit is the input quantum kernel
  @return a `int` with the number of declared qubits.
*/
unsigned int getNumberOfQubits(FuncOp circuit);

/**
 *
 * @param circuit
 * @return
 */
int getNumberOfAllocations(FuncOp circuit);

/**
 * Returns the depth of a quantum circuit.
 * @param circuit The quantum circuit to return the depth of.
 * @return The depth of the quantum circuit passed as argument.
 */
unsigned int getCircuitDepth(FuncOp circuit);

/**
 * Returns the number of gates/instructions of a quantum circuit.
 * @param circuit The quantum circuit to return the number of instructions of.
 * @return The number of instructions of the quantum circuit passed as argument.
 */
unsigned int getNumberOfGates(FuncOp circuit);

/**
  @brief Function that get the number of classical bits used by a given quantum
  kernel.
  @details Given a `FuncOp` that stores a quantum kernel in Quake. This
  function returns the number of declared classical bits within the given
  quantum kernel.
  @param[in] circuit is the input quantum kernel
  @param[out] measurements is a `std::map<int, int>`. This map maps the qubits
  with its corresponding classical bit. The key is the qubit index and the value
  is the classical bit index.
  @return the number of classical bits declared in the given quantum kernel.
*/
int getNumberOfClassicalBits(FuncOp circuit, std::map<int, int> &measurements);

/**
  @brief Function that get the number of classical bits used by a given quantum
  kernel.
  @details Given a `FuncOp` that stores a quantum kernel in Quake. This
  function returns the number of declared classical bits within the given
  quantum kernel.
  @param[in] circuit is the input quantum kernel
  @return the number of classical bits declared in the given quantum kernel.
*/
int getNumberOfClassicalBits(FuncOp circuit);

/**
  @brief Function that get a vector of indices associated with a given
  `ValueRange`.
  @details Given a `ValueRange` that stores a list of indices. This
  function converts the `ValueRange` to a vector of `int`.
  @param[in]  array is the input `ValueRange`.
  @return a vector of indices stored in the input `ValueRange` object.
*/
std::vector<int> getIndicesOfValueRange(ValueRange array);

/**
  @brief Function that get a vector of numerical values associated with a given
  `ValueRange`.
  @details Given a `ValueRange` that stores a list of parameters, i.e.,
  rotation angles. This function converts the `ValueRange` to a vector of
  `double`.
  @param[in]  array is the input `ValueRange` containing the parameters.
  @return a vector of double stored in the input `ValueRange` object.
*/
std::vector<double> getParametersValues(ValueRange array);

/**
  @brief Function get the previous operation on a given target qubit.
  @details Given a `Operation` and a target qubit. This function get the
  previous operation on the given target qubit, starting from `currentOp`.
  @param[in] currentOp is current quantum gate.
  @param[in] targetQubit is the target qubit to be used as reference.
  @return an Operation which is the previous operation on the given target
  qubit.
*/
Operation *getPreviousOperationOnTarget(Operation *currentOp,
                                        Value targetQubit);

/**
  @brief Function get the next operation on a given target qubit.
  @details Given a `Operation` and a target qubit. This function get the
  next operation on the given target qubit, starting from `currentOp`.
  @param[in] currentOp is current quantum gate.
  @param[in] targetQubit is the target qubit to be used as reference.
  @return an Operation which is the next operation on the given target
  qubit.
*/
Operation *getNextOperationOnTarget(Operation *currentOp, Value targetQubit);
} // namespace mqss::support::quakeDialect
#endif // MLIR_UTILS_HPP
