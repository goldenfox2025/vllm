#include <cuda_runtime.h>

#include <cstdio>
#include <exception>

#include "lora_expand_kernel.h"
#include "lora_shrink_kernel.h"
#include "lora_fused_expand_kernel.h"

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

    int input_stride0, int lora_stride0, int lora_stride1, int lora_stride2,
    int output_stride0, int output_stride1, int output_stride2,
    void* stream_ptr, int input_dtype, int output_dtype) {
  try {
    cudaStream_t stream = 0;
    cudaError_t err = cudaStreamSynchronize(stream);

    launch_lora_shrink_kernel(
        input_ptr, lora_a_ptr, output_ptr, token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        max_active_loras, num_total_tokens, hidden_size, lora_rank, num_slices,
        scale, input_stride0, lora_stride0, lora_stride1, lora_stride2,
        output_stride0, output_stride1, output_stride2, stream, input_dtype,
        output_dtype);

    cudaError_t sync_error = cudaStreamSynchronize(stream);
    if (sync_error != cudaSuccess) {
      return -3;
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

// LoRA Fused Expand C interface - 专门处理QKV+LoRA融合计算
int cuda_lora_fused_expand_c(
    void* fused_shrink_input_ptr,            // 融合shrink结果 [num_tokens, total_lora_rank]
    void* lora_b_ptr_array,                  // LoRA B权重指针数组
    void* output_ptr,                        // 输出 [num_tokens, total_hidden_size]
    int* token_indices_sorted_ptr,           // 排序的token索引
    int* lora_ids_ptr,                       // 活跃LoRA ID列表
    int* num_tokens_per_lora_ptr,            // 每个LoRA的token数量
    int* lora_token_start_loc_ptr,           // 每个LoRA的token起始位置
    int* slice_starts_ptr,                   // 每个slice在输出中的起始位置
    int* slice_ranks_ptr,                    // 每个slice的rank
    int* slice_rank_starts_ptr,              // 每个slice在total_lora_rank中的起始位置
    int* lora_strides_d0_ptr,                // LoRA B权重stride
    int* lora_strides_d1_ptr,
    int* lora_strides_d2_ptr,
    int* hidden_sizes_ptr,                   // 每个slice的hidden_size
    int max_active_loras,                    // 最大活跃LoRA数量
    int num_total_tokens,                    // 总token数量
    int total_hidden_size,                   // 总hidden_size
    int num_slices,                          // slice数量
    int add_inputs,                          // 是否累加输入 (0=false, 1=true)
    int fused_input_stride0,                 // 融合输入的stride
    int fused_input_stride1,
    int output_stride0,                      // 输出的stride
    int output_stride1,
    void* stream_ptr,                        // CUDA流
    int input_dtype,                         // 输入数据类型
    int output_dtype                         // 输出数据类型
) {
    cudaStream_t stream = 0;  // 使用默认流
    
    try {
        launch_lora_fused_expand_kernel(
            fused_shrink_input_ptr, lora_b_ptr_array, output_ptr,
            token_indices_sorted_ptr, lora_ids_ptr, num_tokens_per_lora_ptr,
            lora_token_start_loc_ptr, slice_starts_ptr, slice_ranks_ptr,
            slice_rank_starts_ptr, lora_strides_d0_ptr, lora_strides_d1_ptr,
            lora_strides_d2_ptr, hidden_sizes_ptr, max_active_loras,
            num_total_tokens, total_hidden_size, num_slices, (bool)add_inputs,
            fused_input_stride0, fused_input_stride1, output_stride0, output_stride1,
            stream, input_dtype, output_dtype
        );

        // 同步检查错误
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            printf("CUDA fused expand kernel execution error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        return 0;  // 成功
    } catch (...) {
        printf("Exception in CUDA fused expand kernel\n");
        return 1;  // 错误
    }
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
