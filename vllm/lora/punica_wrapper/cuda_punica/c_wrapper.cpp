#include <cuda_runtime.h>

#include <cstdio>
#include <exception>

#include "lora_expand_kernel.h"
#include "lora_shrink_kernel.h"

extern "C" {

int cuda_lora_shrink_c(
    void* input_ptr,                // Input data pointer
    void* lora_a_ptr,               // LoRA A weights pointer
    void* output_ptr,               // Output data pointer
    int* token_indices_sorted_ptr,  // Token indices sorted by LoRA IDs
    int* lora_ids_ptr,              // Active LoRA IDs array
    int* num_tokens_per_lora_ptr,   // Number of tokens per LoRA
    int* lora_token_start_loc_ptr,  // Start location for each LoRA
    int max_active_loras,           // Number of active LoRAs
    int num_total_tokens, int hidden_size, int lora_rank, int num_slices,
    float scale,
    // Strides
    int input_stride0, int lora_stride0, int lora_stride1, int lora_stride2,
    int output_stride0, int output_stride1, int output_stride2,
    void* stream_ptr, int input_dtype, int output_dtype) {
  try {
    // Use default CUDA stream (0) for simplicity
    // This is compatible with most PyTorch operations
    cudaStream_t stream = 0;
    cudaError_t err = cudaStreamSynchronize(stream);

    // Launch the kernel with dtype information and multi-LoRA support
    launch_lora_shrink_kernel(
        input_ptr, lora_a_ptr, output_ptr, token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        max_active_loras, num_total_tokens, hidden_size, lora_rank, num_slices,
        scale, input_stride0, lora_stride0, lora_stride1, lora_stride2,
        output_stride0, output_stride1, output_stride2, stream, input_dtype,
        output_dtype);

    // Ensure kernel completion
    cudaError_t sync_error = cudaStreamSynchronize(stream);
    if (sync_error != cudaSuccess) {
      return -3;  // Synchronization error
    }

    return 0;  // Success
  } catch (...) {
    return -1;  // Error
  }
}

// LoRA Expand C interface
int cuda_lora_expand_c(
    void* input_ptr,   // Input data pointer [num_slices, num_tokens, lora_rank]
    void* lora_b_ptr,  // 指针数组，每个元素指向一个slice的LoRA权重
    void* output_ptr,  // Output data pointer [num_tokens, hidden_size *
                       // num_slices]
    int* token_indices_sorted_ptr,  // Token indices sorted by LoRA [num_tokens]
    int* lora_ids_ptr,              // LoRA IDs [max_loras]
    int* num_tokens_per_lora_ptr,   // Number of tokens per LoRA [max_loras]
    int* lora_token_start_loc_ptr,  // LoRA token start locations [max_loras+1]
    int* slice_starts_ptr,          // Slice start positions [num_slices]
    int* lora_strides_d0_ptr,       // LoRA strides d0 [num_slices]
    int* lora_strides_d1_ptr,       // LoRA strides d1 [num_slices]
    int* lora_strides_d2_ptr,       // LoRA strides d2 [num_slices]
    int* hidden_sizes_ptr,          // Hidden sizes [num_slices]
    int max_active_loras,           // Maximum active LoRAs
    int num_total_tokens_in_batch,  // Total tokens in batch
    int lora_rank, int hidden_size, int num_slices, int offset_start,
    int add_inputs,  // add_inputs: 0=false, 1=true
    // Strides
    int input_stride0, int input_stride1, int input_stride2, int output_stride0,
    int output_stride1, void* stream_ptr, int input_dtype, int output_dtype) {
  // Use default CUDA stream (0) for simplicity
  cudaStream_t stream = 0;
  cudaError_t err = cudaStreamSynchronize(stream);
  try {
    // Launch the clean expand kernel with new Triton-compatible interface
    launch_lora_expand_kernel(
        input_ptr, lora_b_ptr, output_ptr, token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        slice_starts_ptr,  // 新增参数
        lora_strides_d0_ptr, lora_strides_d1_ptr,
        lora_strides_d2_ptr,  // 新增参数
        hidden_sizes_ptr,     // 新增参数
        max_active_loras, num_total_tokens_in_batch, lora_rank, hidden_size,
        num_slices, offset_start, (bool)add_inputs, input_stride0,
        input_stride1, input_stride2, output_stride0, output_stride1, stream,
        input_dtype, output_dtype);

    // Synchronize to catch any kernel errors
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      printf("CUDA expand kernel execution error: %s\n",
             cudaGetErrorString(err));
      return 1;  // Error
    }

  } catch (...) {
    printf("Exception in CUDA expand kernel\n");
    return 1;  // Error
  }

  return 0;  // Success
}

/**
 * Simple test function that can be called from Python
 */
int test_cuda_kernel() {
  // Simple test to verify the library loads
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? 0 : -1;
}

}  // extern "C"
