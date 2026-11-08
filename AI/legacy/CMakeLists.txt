# Legacy, gate-pattern MLIR pass library used by the AI pass selector.
#
# The upstream compilation suite replaced these one-rewrite-per-pass Quake
# passes with dialect-agnostic "Common*" passes (see lib/Passes/Transforms).
# The reinforcement-learning agents in AI/ were trained on the action space
# defined by the passes below, so they are kept here, self-contained, and are
# only built when BUILD_MQSS_AI is ON.
#
# Targets: SupportQuake, Interfaces, MLIRPasses, Optimizer (+ MQSS:: aliases).

set(MQSS_AI_LEGACY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(MQSS_AI_LEGACY_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

file(MAKE_DIRECTORY ${MQSS_AI_LEGACY_BINARY_DIR}/include/Passes)

# TableGen pass declarations.
set(LLVM_TARGET_DEFINITIONS ${MQSS_AI_LEGACY_DIR}/include/Passes/Transforms.td)
mlir_tablegen(include/Passes/Transforms.h.inc -gen-pass-decls -name
              MQSSOptTransforms)
add_public_tablegen_target(MQSSAILegacyTransformsPassIncGen)

set(LLVM_TARGET_DEFINITIONS
    ${MQSS_AI_LEGACY_DIR}/include/Passes/Decompositions.td)
mlir_tablegen(include/Passes/Decompositions.h.inc -gen-pass-decls -name
              MQSSOptDecompositions)
add_public_tablegen_target(MQSSAILegacyDecompositionsPassIncGen)

set(LLVM_TARGET_DEFINITIONS
    ${MQSS_AI_LEGACY_DIR}/include/Passes/Cancellations.td)
mlir_tablegen(include/Passes/Cancellations.h.inc -gen-pass-decls -name
              MQSSOptCancellations)
add_public_tablegen_target(MQSSAILegacyCancellationsPassIncGen)

set(MQSS_AI_LEGACY_INCLUDE_DIRS
    ${MQSS_AI_LEGACY_DIR}/include
    ${MQSS_AI_LEGACY_DIR}/include/Passes
    ${MQSS_AI_LEGACY_DIR}/include/Support
    ${MQSS_AI_LEGACY_DIR}/include/Support/DAG
    ${MQSS_AI_LEGACY_DIR}/include/Optimizer
    ${MQSS_AI_LEGACY_BINARY_DIR}/include
    ${MQT_QMAP_INCLUDE_DIR}
    ${MQT_CORE_INCLUDE_DIR}
    ${CUDAQ_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS})

file(GLOB SRC_TRANSFORMS "${MQSS_AI_LEGACY_DIR}/lib/Passes/Transforms/*.cpp")
file(GLOB SRC_DECOMPOSITIONS
     "${MQSS_AI_LEGACY_DIR}/lib/Passes/Decompositions/*.cpp")
file(GLOB SRC_CANCELLATIONS
     "${MQSS_AI_LEGACY_DIR}/lib/Passes/Cancellations/*.cpp")
file(GLOB SRC_CODEGEN "${MQSS_AI_LEGACY_DIR}/lib/Passes/CodeGen/*.cpp")
file(GLOB SRC_EXAMPLES "${MQSS_AI_LEGACY_DIR}/lib/Passes/Examples/*.cpp")
file(GLOB SRC_OPTIMIZER "${MQSS_AI_LEGACY_DIR}/lib/Optimizer/*.cpp")
file(GLOB SRC_INTERFACES "${MQSS_AI_LEGACY_DIR}/lib/Interfaces/*.cpp")
file(GLOB_RECURSE SRC_SUPPORT_QUAKE "${MQSS_AI_LEGACY_DIR}/lib/Support/*.cpp")

set(MQSS_AI_CUDAQ_LIBS ${CUDAQ_QUAKE_LIBRARY} ${CUDAQ_CC_LIBRARY})

add_mlir_library(
  SupportQuake
  SHARED
  ${SRC_SUPPORT_QUAKE}
  LINK_LIBS
  PUBLIC
  ${MQSS_AI_CUDAQ_LIBS}
  MLIRIR
  MLIRPass
  MLIRTransforms)

add_mlir_library(
  Interfaces
  SHARED
  ${SRC_INTERFACES}
  LINK_LIBS
  PUBLIC
  SupportQuake
  MQT::CoreIR
  MQT::QMapSCHeuristic
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRPass
  MLIRTransforms)

add_mlir_library(
  MLIRPasses
  SHARED
  ${SRC_TRANSFORMS}
  ${SRC_DECOMPOSITIONS}
  ${SRC_CANCELLATIONS}
  ${SRC_CODEGEN}
  ${SRC_EXAMPLES}
  DEPENDS
  MQSSAILegacyTransformsPassIncGen
  MQSSAILegacyDecompositionsPassIncGen
  MQSSAILegacyCancellationsPassIncGen
  LINK_LIBS
  PUBLIC
  SupportQuake
  Interfaces
  MQT::CoreIR
  MQT::QMapSCHeuristic
  MLIRTensorDialect
  MLIRIR
  MLIRPass
  MLIRTransforms)

add_mlir_library(
  Optimizer
  SHARED
  ${SRC_OPTIMIZER}
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPasses)

foreach(tgt SupportQuake Interfaces MLIRPasses Optimizer)
  # Legacy headers must win over the upstream include/ tree.
  target_include_directories(${tgt} BEFORE PUBLIC ${MQSS_AI_LEGACY_INCLUDE_DIRS})
  target_compile_options(${tgt} PRIVATE -fexceptions -frtti)
  if(TARGET obj.${tgt})
    target_include_directories(obj.${tgt} BEFORE
                               PUBLIC ${MQSS_AI_LEGACY_INCLUDE_DIRS})
    target_compile_options(obj.${tgt} PRIVATE -fexceptions -frtti)
  endif()
  add_library(MQSS::${tgt} ALIAS ${tgt})
endforeach()
