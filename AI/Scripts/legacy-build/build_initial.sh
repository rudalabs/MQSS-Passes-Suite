#!/bin/bash

# Clear the terminal screen
clear
git config --global --add safe.directory '*'
CURRENT_DIR=$(pwd)

# Defaults
INSTALL_PATH="${INSTALL_PATH:-$HOME/.passes}"
NUM_JOBS=1
BUILD_DOCS=OFF
BUILD_TESTS=ON
BUILD_TOOLS=ON
BUILD_AI=ON
BUILD_TYPE="Release"
MLIR_DIR="${MLIR_DIR:-/usr/local/llvm/lib/cmake/mlir}"
CLANG_DIR="${CLANG_DIR:-/usr/local/llvm/lib/cmake/clang}"
LLVM_DIR="${LLVM_DIR:-/usr/local/llvm/lib/cmake/llvm}"
INSTALL_DIR="${INSTALL_PATH:-$HOME/.passes}"

# Default directories (can be overridden by arguments)
MLIR_DIR="${MLIR_DIR:-/usr/local/llvm/lib/cmake/mlir}"
CLANG_DIR="${CLANG_DIR:-/usr/local/llvm/lib/cmake/clang}"
LLVM_DIR="${LLVM_DIR:-/usr/local/llvm/lib/cmake/llvm}"
INSTALL_DIR="${INSTALL_PATH:-$HOME/.passes}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -j|--jobs) NUM_JOBS="$2"; shift 2;;
    --debug) BUILD_TYPE="Debug"; shift;;
    --mlir-dir) MLIR_DIR="$2"; shift 2;;
    --install-dir) INSTALL_DIR="$2"; shift 2;;
    --clang-dir) CLANG_DIR="$2"; shift 2;;
    --llvm-dir) LLVM_DIR="$2"; shift 2;;
    --build-tools) BUILD_TOOLS=ON; shift;;
    --build-docs) BUILD_DOCS=ON; shift;;
    --build-tests) BUILD_TESTS=ON; shift;;
    --build-ai) BUILD_AI=ON; shift;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

# --- detect CUDA via HPC-SDK module ---
CUDA_ENABLED=0
CUDA_URL_SUFFIX="nightly/cpu"
CUDA_CMAKE_ARGS=()

if module use /opt/nvidia/hpc_sdk/modulefiles && module load nvhpc/25.5 && command -v nvcc >/dev/null 2>&1; then
  # HPC-SDK has versioned CUDA roots; prefer 12.9 if present, else pick newest
  HPC_BASE="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda"
  if [[ -d "${HPC_BASE}/12.9" ]]; then
    CUDA_HOME="${HPC_BASE}/12.9"
  else
    # pick highest version that has targets dir
    CUDA_HOME="$(ls -d ${HPC_BASE}/* 2>/dev/null | grep -E '/[0-9]+\.[0-9]+$' | sort -V | tail -1)"
  fi

  if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/targets/x86_64-linux/include" ]]; then
    CUDA_ENABLED=1
    CUDA_URL_SUFFIX="test/cu129"   # for LibTorch with CUDA 12.9
    NVCC="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin/nvcc"
    CUDA_INCLUDE_DIRS="${CUDA_HOME}/targets/x86_64-linux/include"
    CUDA_CUDART_LIBRARY="${CUDA_HOME}/targets/x86_64-linux/lib/libcudart.so"

    CUDA_CMAKE_ARGS+=(
      "-DCUDAToolkit_ROOT=${CUDA_HOME}"
      "-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}"
      "-DCUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS}"
      "-DCUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY}"
      "-DCMAKE_CUDA_COMPILER=${NVCC}"
      "-DCUDA_NVCC_EXECUTABLE=${NVCC}"
      "-DCMAKE_POLICY_VERSION_MINIMUM=3.10"
      "-DCUDA_NVRTC_LIBRARY=${CUDA_HOME}/targets/x86_64-linux/lib/libnvrtc.so"
    )
    echo "[CUDA] enabled via HPC-SDK at ${CUDA_HOME}"
  else
    echo "[CUDA] HPC-SDK present but headers/libs not found; proceeding CPU-only."
  fi
else
  echo "[CUDA] modules/nvcc not available; proceeding CPU-only."
fi

########################################################################################################################
# Build external tools necessary for the AI submodule
AI_DIR=${CURRENT_DIR}"/AI"
AI_EXTERNAL_DIR=${AI_DIR}"/external"
LIBTORCH_DIR=${AI_EXTERNAL_DIR}"/libtorch"
mkdir -p "${AI_EXTERNAL_DIR}"
if [[ ! -d "${LIBTORCH_DIR}" ]]; then
  pushd "${AI_EXTERNAL_DIR}" >/dev/null
  echo "After pushd."
  LIBTORCH_ZIP="libtorch-shared-with-deps-latest.zip"
  LIBTORCH_URL="https://download.pytorch.org/libtorch/${CUDA_URL_SUFFIX}/${LIBTORCH_ZIP}"
  echo "[LibTorch] fetching ${LIBTORCH_URL}"
  wget -O "${LIBTORCH_ZIP}" "${LIBTORCH_URL}"
  unzip "${LIBTORCH_ZIP}"
  rm -rfv "${LIBTORCH_ZIP}"
  popd >/dev/null
else
  echo "[LibTorch] exists at ${LIBTORCH_DIR}"
fi
cd "${CURRENT_DIR}"
########################################################################################################################

BUILD_DIR=${CURRENT_DIR}"/build"
DEPS_DIR="${BUILD_DIR}/_deps"
CUDAQ_DIR="${DEPS_DIR}/cuda-quantum"
CUDAQ_REPO="https://github.com/NVIDIA/cuda-quantum.git"

# Create directories if they don't exist
mkdir -p "${BUILD_DIR}"
mkdir -p "${DEPS_DIR}"

# Clone the CUDA Quantum repository
echo "Cloning CUDA Quantum repository into ${CUDAQ_DIR}."
if [ ! -d "${CUDAQ_DIR}" ]; then
  git clone "${CUDAQ_REPO}" "${CUDAQ_DIR}"
  if [ $? -ne 0 ]; then
    echo "Failed to clone CUDA Quantum repository."
    exit 1
  fi
else
  echo "CUDA Quantum repository already exists at ${CUDAQ_DIR}. Skipping clone."
fi

# Navigate to the CUDA Quantum directory
cd "${CUDAQ_DIR}" || { echo "Failed to navigate to ${CUDAQ_DIR}."; exit 1; }

# Create a build directory
mkdir -p build && cd build || { echo "Failed to create or navigate to build directory."; exit 1; }

# Configure CUDA Quantum using CMake
echo "Configuring CUDA Quantum with CMake."
cmake -G Ninja \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DClang_DIR="${CLANG_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
	-DCUDAQ_BUILD_TESTS=OFF \
  -DCUDAQ_ENABLE_REST=ON \
  -DOPENSSL_ROOT_DIR=/usr \
  -DOPENSSL_INCLUDE_DIR=/usr/include \
  -DOPENSSL_SSL_LIBRARY=/lib64/libssl.so \
  -DOPENSSL_CRYPTO_LIBRARY=/lib64/libcrypto.so \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
  ..

if [ $? -ne 0 ]; then
  echo "CMake configuration failed."
  exit 1
fi

# Build the cudaq-mlir-runtime target using Ninja
echo "Building cudaq-mlir-runtime target with ${NUM_JOBS} jobs."
ninja -j"${NUM_JOBS}" cudaq-mlir-runtime

if [ $? -ne 0 ]; then
  echo "Failed to build cudaq-mlir-runtime target."
  exit 1
fi

echo "Build completed successfully!"

echo ${BUILD_DIR}
cd  "${BUILD_DIR}" || { echo "Failed to navigate back to the original directory."; exit 1; }

echo "Configuring MQSS Passes Repository CMake."
cmake .. \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DBUILD_MLIR_PASSES_TOOLS="${BUILD_TOOLS}" \
  -DBUILD_MLIR_PASSES_DOCS="${BUILD_DOCS}" \
  -DBUILD_MLIR_PASSES_TESTS="${BUILD_TESTS}"\
  -DBUILD_MLIR_PASSES_AI="${BUILD_AI}" \
  -DCUDAQ_SOURCE_DIR="${CUDAQ_DIR}" \
	-DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_BUILD_RPATH="${HOME}/.local/lib;${HOME}/.local/lib64" \
  -DCMAKE_INSTALL_RPATH="${HOME}/.local/lib;${HOME}/.local/lib64" \
  -DTorch_DIR="${LIBTORCH_DIR}" \
  -U CMAKE_CUDA_ARCHITECTURES \
  -DTORCH_CUDA_ARCH_LIST="8.0" \
  -DCUDA_NVRTC_LIBRARY="${CUDA_HOME}/targets/x86_64-linux/lib/libnvrtc.so" \
  "${CUDA_CMAKE_ARGS[@]}"

echo "Building MQSS Repository Passes with ${NUM_JOBS} jobs."
make -j"${NUM_JOBS}"
echo "Build of MQSS Repository Passes completed!"
