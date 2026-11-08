#!/usr/bin/env bash

# Clear console log
set -euo pipefail
clear

# Rebuild MQSS only. Assumes: AI/external/libtorch + tensorflow + _deps/cuda-quantum built.
CURRENT_DIR="$(pwd)"
NUM_JOBS="4"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# Optional arg parsing: -j/--jobs, --debug only
while [[ $# -gt 0 ]]; do
  case $1 in
    -j|--jobs) NUM_JOBS="$2"; shift 2;;
    --debug) BUILD_TYPE="Debug"; shift;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

# Required dirs (reuse env if set)
MLIR_DIR="${MLIR_DIR:-/usr/local/llvm/lib/cmake/mlir}"
LLVM_DIR="${LLVM_DIR:-/usr/local/llvm/lib/cmake/llvm}"
INSTALL_DIR="${INSTALL_PATH:-$HOME/.passes}"
BUILD_DIR="${CURRENT_DIR}/build"
CUDAQ_DIR="${BUILD_DIR}/_deps/cuda-quantum"

# Torch dir
LIBTORCH_DIR=${CURRENT_DIR}"/AI/external/libtorch"

# Detect CUDA (same logic)
CUDA_ENABLED=0
CUDA_CMAKE_ARGS=()
if module use /opt/nvidia/hpc_sdk/modulefiles && module load nvhpc/25.5 && command -v nvcc >/dev/null 2>&1; then
  HPC_BASE="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda"
  if [[ -d "${HPC_BASE}/12.9" ]]; then
    CUDA_HOME="${HPC_BASE}/12.9"
  else
    CUDA_HOME="$(ls -d ${HPC_BASE}/* 2>/dev/null | grep -E '/[0-9]+\.[0-9]+$' | sort -V | tail -1)"
  fi
  if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/targets/x86_64-linux/include" ]]; then
    export CAFFE2_NVRTC_LIBRARY="${HPC_BASE}/12.9/targets/x86_64-linux/lib/stubs/libnvrtc.so"
    CUDA_ENABLED=1
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
    echo "[CUDA] HPC-SDK present but headers/libs not found; building CPU-only."
  fi
else
  echo "[CUDA] modules/nvcc not available; building CPU-only."
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Reconfiguring MQSS (rebuild)..."
CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
  -DMLIR_DIR="${MLIR_DIR}"
  -DLLVM_DIR="${LLVM_DIR}"
	-DCUDAQ_BUILD_TESTS=ON
  -DBUILD_MLIR_PASSES_AI=ON
  -DCUDAQ_SOURCE_DIR="${CUDAQ_DIR}"
  -DCMAKE_BUILD_RPATH="${HOME}/.local/lib;${HOME}/.local/lib64"
  -DCMAKE_INSTALL_RPATH="${HOME}/.local/lib;${HOME}/.local/lib64"
  -DTorch_DIR="${LIBTORCH_DIR}"
  -U CMAKE_CUDA_ARCHITECTURES
  -DTORCH_CUDA_ARCH_LIST="8.0"
  -DCMAKE_MESSAGE_LOG_LEVEL=ERROR
)

if [[ $CUDA_ENABLED -eq 1 ]]; then
  CMAKE_ARGS+=(
    -DCUDA_NVRTC_LIBRARY="${CUDA_HOME}/targets/x86_64-linux/lib/libnvrtc.so"
    -DCUDA_NVRTC_INCLUDE_DIR="${CUDA_HOME}/targets/x86_64-linux/include"
    "${CUDA_CMAKE_ARGS[@]}"
  )
fi

cmake .. "${CMAKE_ARGS[@]}"

echo "Building with ${NUM_JOBS} jobs..."
make -j"${NUM_JOBS}"
echo "Rebuild of MQSS Repository Passes completed!"
