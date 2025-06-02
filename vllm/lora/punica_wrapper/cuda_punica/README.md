# CUDA LoRA Kernel for vLLM

This directory contains a CUDA implementation of LoRA operations for vLLM, designed to replace the Triton-based kernels with optimized CUDA code.

## Overview

The LoRA (Low-Rank Adaptation) computation consists of two main operations:
1. **Shrink**: `buffer = input @ lora_a * scale` (hidden_size → lora_rank)
2. **Expand**: `output += buffer @ lora_b + bias` (lora_rank → hidden_size)

This implementation focuses on the **shrink operation** as a starting point, with plans to fuse both operations for optimal performance.

## Mathematical Background

LoRA computation: `y = x @ W + x @ (A @ B) * scale`

Where:
- `x`: Input tensor [num_tokens, hidden_size]
- `W`: Original weight matrix
- `A`: LoRA down-projection [lora_rank, hidden_size] 
- `B`: LoRA up-projection [hidden_size, lora_rank]
- `scale`: Scaling factor

The two-step process:
1. **Shrink**: `buffer = x @ A^T * scale` 
2. **Expand**: `y += buffer @ B^T + bias`

## File Structure

```
cuda_punica/
├── CMakeLists.txt              # CMake build configuration
├── build.sh                   # Build script
├── lora_shrink_kernel.h        # CUDA kernel header
├── lora_shrink_kernel.cu       # CUDA kernel implementation
├── pybind_wrapper.cpp          # PyBind11 Python bindings
├── punica_gpu_cuda_integration.py  # Integration example
└── README.md                   # This file
```

## Building

### Prerequisites

- CUDA Toolkit (11.0+)
- CMake (3.18+)
- PyBind11
- PyTorch with CUDA support

### Build Steps

```bash
# Navigate to the cuda_punica directory
cd vllm/lora/punica_wrapper/cuda_punica

# Run the build script
./build.sh
```

The build script will:
1. Find PyBind11 automatically
2. Configure CMake with appropriate CUDA architectures
3. Compile the CUDA kernel and Python bindings
4. Generate `cuda_punica.so` extension module

### Manual Build

```bash
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"

make -j$(nproc)
```

## Testing

### Simple Test (No Profiling)

```bash
# From vLLM root directory
./test_cuda_lora_simple.sh
```

This script:
- Builds the CUDA kernel
- Runs correctness tests against PyTorch reference
- Tests multiple LoRA configurations
- Performs basic performance benchmarking

### NCU Profiling Test

```bash
# From vLLM root directory  
./test_cuda_lora_ncu.sh
```

This script:
- Builds the CUDA kernel
- Runs the kernel with NVIDIA Nsight Compute profiling
- Generates detailed performance reports
- Saves reports to `ncu_reports/` directory

### View NCU Reports

```bash
# Open in Nsight Compute UI
ncu-ui ncu_reports/cuda_lora_TIMESTAMP.ncu-rep

# Generate text summary
ncu --import ncu_reports/cuda_lora_TIMESTAMP.ncu-rep --page details
```

## Integration with vLLM

### Current Status

The CUDA kernel is implemented as a **drop-in replacement** for the Triton `lora_shrink` operation. 

### Integration Steps

1. **Build the kernel** using `build.sh`

2. **Modify punica_gpu.py** to use CUDA kernel:

```python
# Add import
try:
    from .cuda_punica import cuda_punica
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Modify add_shrink method
def add_shrink(self, y, x, lora_a_stacked, scale, **kwargs):
    x = x.view(-1, x.shape[-1])
    
    if CUDA_AVAILABLE and self.use_cuda_kernel:
        # Use CUDA kernel
        token_lora_indices = self._extract_token_lora_mapping(x.size(0))
        cuda_punica.lora_shrink(x, lora_a_stacked, y, token_lora_indices, scale)
    else:
        # Fallback to Triton
        lora_shrink(x, lora_a_stacked, y, 
                   *self.token_mapping_meta.meta_args(x.size(0)), scale)
```

3. **Enable CUDA kernel** via environment variable:

```bash
export VLLM_USE_CUDA_LORA_KERNEL=1
```

### Future Work: Fused Kernel

The ultimate goal is to create a **fused shrink+expand kernel** that:
- Eliminates intermediate buffer allocation
- Reduces memory bandwidth requirements  
- Minimizes kernel launch overhead
- Optimizes register and shared memory usage

This would be integrated at the `add_lora_linear` level in `punica_gpu.py`.

## Performance Considerations

### Current Implementation

- **Tiled matrix multiplication** with shared memory
- **Coalesced memory access** patterns
- **Register blocking** for accumulation
- **Multi-LoRA support** with token-to-LoRA mapping

### Optimization Opportunities

1. **Tensor Core utilization** for mixed-precision computation
2. **Warp-level primitives** for better instruction throughput
3. **Memory access optimization** for different tensor layouts
4. **Kernel fusion** to eliminate intermediate buffers

## Debugging

### Common Issues

1. **Build failures**: Check CUDA toolkit installation and CMake version
2. **Import errors**: Ensure `cuda_punica.so` is in the correct location
3. **Runtime errors**: Verify tensor shapes and data types match expectations

### Debug Build

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### CUDA Error Checking

The kernel includes error checking:
```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
}
```

## Contributing

When modifying the CUDA kernel:

1. **Test thoroughly** with different tensor sizes and LoRA configurations
2. **Profile performance** using NCU to identify bottlenecks
3. **Verify correctness** against the Triton reference implementation
4. **Update documentation** for any interface changes

## References

- [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyBind11 Documentation](https://pybind11.readthedocs.io/)
- [vLLM Documentation](https://docs.vllm.ai/)
