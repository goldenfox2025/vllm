cmake_minimum_required(VERSION 3.18)
project(cuda_punica LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED) # Found for potential future use in Python bindings
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Set CUDA properties
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture specifically to sm_89.
set(CMAKE_CUDA_ARCHITECTURES "89")
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Get PyTorch paths using torch.utils.cpp_extension
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.library_paths()))"
    OUTPUT_VARIABLE TORCH_LIBRARY_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get PyTorch CMake prefix path to help find_package(Torch)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find PyTorch
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # For cuda_runtime.h etc.
include_directories(${Python3_INCLUDE_DIRS})                 # For Python.h if c_wrapper directly uses it
include_directories(${TORCH_INCLUDE_PATHS})                 # For torch/extension.h etc.

# Library directories (for linker to find PyTorch libs if linked by name)
link_directories(${TORCH_LIBRARY_PATHS})

# Compiler flags for optimization and settings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -O3 \
    -Xptxas -O3 \
    -use_fast_math \
    -lineinfo \
    -Xcompiler -Wall,-Wextra,-Wpedantic,-DNDEBUG \
    ")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -O3 \
    -fPIC \
    -Wall \
    -Wextra \
    -Wpedantic \
    -ffast-math \
    -march=native \
    -DNDEBUG \
    ")
# Note: -march=native optimizes for the build machine's CPU.
# For broader compatibility, consider a specific architecture or remove.

# Disable LTO (as per original requirement)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Source files
set(CUDA_SOURCES
    lora_shrink_kernel.cu
    lora_expand_kernel.cu
)
set(C_SOURCES
    c_wrapper.cpp
)

# Create the shared library for CUDA/C++ code
add_library(cuda_lora_c SHARED ${CUDA_SOURCES} ${C_SOURCES})

# Set properties for the library
set_target_properties(cuda_lora_c PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    # OUTPUT_NAME "cuda_lora_c" # Optional: define library name without "lib" prefix
)

# Link necessary libraries to cuda_lora_c
# For this "pure C" library, only CUDA libraries are directly linked.
# PyTorch (Torch::torch) is not linked here, assuming c_wrapper.cpp and CUDA kernels
# do not directly depend on PyTorch's C++ API for this target.
target_link_libraries(cuda_lora_c PRIVATE ${CUDA_LIBRARIES})

# --- Pybind11 Integration Notes ---
# If c_wrapper.cpp were a pybind11 wrapper, you would:
# 1. Add pybind11_INCLUDE_DIRS to include_directories.
# 2. Link Python3::Python, pybind11::module (or pybind11::embed), and potentially Torch::torch
#    to a pybind11_add_module target.
# For now, pybind11 is found but not actively used for the cuda_lora_c SHARED target.

# Informational messages (useful for debugging build configurations)
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "PyTorch include paths: ${TORCH_INCLUDE_PATHS}")
message(STATUS "PyTorch library paths: ${TORCH_LIBRARY_PATHS}")
message(STATUS "Torch_LIBRARIES (from find_package(Torch)): ${Torch_LIBRARIES}")
message(STATUS "CUDA_LIBRARIES (from find_package(CUDA)): ${CUDA_LIBRARIES}")

