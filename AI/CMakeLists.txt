cmake_minimum_required(VERSION 3.19)
project(AIPassSelector LANGUAGES CXX)

# Vendored SDK discovery roots
list(APPEND CMAKE_PREFIX_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch"
)

# Find each SDK, but we will not mix them in a single target
find_package(Torch REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Legacy gate-pattern pass library that defines the agents' action space.
add_subdirectory(legacy)

# The CUDA-Q MLIR runtime (common/RuntimeMLIR.h) is only available when CUDA-Q
# was built with its runtime enabled.
set(MQSS_AI_CUDAQ_RUNTIME "")
if (TARGET cudaq-mlir-runtime)
    set(MQSS_AI_CUDAQ_RUNTIME cudaq-mlir-runtime)
else ()
    find_library(MQSS_AI_CUDAQ_RUNTIME_LIB cudaq-mlir-runtime
            HINTS ${CUDAQ_INSTALL_PREFIX}/lib ${CUDAQ_ROOT}/lib)
    if (MQSS_AI_CUDAQ_RUNTIME_LIB)
        set(MQSS_AI_CUDAQ_RUNTIME ${MQSS_AI_CUDAQ_RUNTIME_LIB})
    else ()
        message(WARNING "cudaq-mlir-runtime not found; AI targets may fail to link.")
    endif ()
endif ()

# ---------------- Shared SDK-agnostic ----------------
add_library(common_ai_props INTERFACE)
target_include_directories(common_ai_props INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/legacy/include"
        "${CMAKE_CURRENT_BINARY_DIR}/legacy/include"
)
target_compile_definitions(common_ai_props INTERFACE
        AI_DATASET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Datasets"
        AI_AGENTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Agents"
        AI_CIRCUITS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Circuits"
        RQCG_STATISTICS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include/Environment/StatisticsForRQCG"
        MQSS_BUILD_DIR="${CMAKE_BINARY_DIR}"
)
target_link_libraries(common_ai_props INTERFACE
        MLIRIR
        MLIRTransforms
        ${MQSS_AI_CUDAQ_RUNTIME}
        MLIRPasses
        SupportQuake
        Optimizer
        Interfaces
        yaml-cpp
)
if (NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(common_ai_props INTERFACE -Ofast)
    if (NOT APPLE)
        target_compile_options(common_ai_props INTERFACE -march=native)
    endif ()
endif ()

# Gather sources.
# No SDK headers/sources here.
file(GLOB_RECURSE ALL_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cpp")

add_library(ai_core STATIC ${ALL_HEADERS} ${ALL_SOURCES})
target_link_libraries(ai_core PUBLIC common_ai_props)
target_include_directories(ai_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---------------- Torch targets ----------------
if (TARGET Torch::Torch)
    target_link_libraries(ai_core PUBLIC Torch::Torch)
else ()
    target_link_libraries(ai_core PUBLIC "${TORCH_LIBRARIES}")
    target_include_directories(ai_core PUBLIC "${TORCH_INCLUDE_DIRS}")
endif ()
target_compile_options(ai_core PRIVATE ${TORCH_CXX_FLAGS})

add_executable(ai_pass_selector ai_pass_selector.cpp)
target_link_libraries(ai_pass_selector PRIVATE ai_core yaml-cpp)

# ---------------- Neutral executables ----------------
add_executable(convert_qasm_dataset_to_quake convert_qasm_dataset_to_quake.cpp)
target_link_libraries(convert_qasm_dataset_to_quake PRIVATE ai_core)

add_executable(convert_quake_testsuite_to_tikz convert_quake_testsuite_to_tikz.cpp)
target_link_libraries(convert_quake_testsuite_to_tikz PRIVATE ai_core)
