#!/usr/bin/env bash

# Exit immediately if any command fails
set -e
REPO_DIR=$(dirname "$(readlink -f "$0")")

echo "=== 1. Pulling latest updates from GitHub ==="
git fetch --all
git reset --hard origin/master

echo "=== 2. Cleaning old build files ==="
rm -rf build/

echo "=== 3. Configuring build with CMake ==="
# Default: CPU Only. 
# For NVIDIA: change to -DGGML_CUDA=ON
# For AMD:    change to -DGGML_HIPBLAS=ON
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON

echo "=== 4. Compiling binaries ==="
cmake --build build --config Release -j$(nproc)

sudo ln -sf "${REPO_DIR}/build/bin/llama-server" /usr/local/bin/llama-serve

echo "=== Success! Llama.cpp has been successfully rebuilt ==="
