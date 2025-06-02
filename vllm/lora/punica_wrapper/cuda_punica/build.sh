#!/bin/bash

# Build script for CUDA LoRA kernel
# This script compiles the CUDA kernel and creates a Python extension module

set -e  # Exit on any error

echo "🛠️  Building CUDA LoRA kernel..."

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "🧹 Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "🔍 Finding pybind11..."
PYBIND11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')
echo "📍 pybind11 found at: $PYBIND11_DIR"

echo "🛠️  Running CMake..."
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"

echo "🔨 Compiling..."
make -j$(nproc)

echo "✅ Build completed successfully!"
echo "📦 Extension module: $BUILD_DIR/cuda_punica*.so"

# Copy the built module to parent directory for easy import
# echo "📋 Copying module to parent directory..."
# cp cuda_punica*.so ../

echo "🎉 CUDA LoRA kernel build complete!"
# echo ""
# echo "To test the module:"
# echo "  cd .."
echo "  python3 -c 'import cuda_punica; print(\"✅ Module imported successfully!\")'"
