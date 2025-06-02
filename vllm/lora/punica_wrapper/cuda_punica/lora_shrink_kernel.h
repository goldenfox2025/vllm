#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// CUDA kernel for LoRA shrink operation
// Performs: output = input @ lora_a_weights * scale
// where input: [num_tokens, hidden_size]
//       lora_a_weights: [num_loras, lora_rank, hidden_size]
//       output: [num_slices, num_tokens, lora_rank]

/**
 * CUDA kernel for LoRA shrink operation (x @ A * scale)
 *
 * This kernel performs the first step of LoRA computation:
 * For each token, multiply with the appropriate LoRA A matrix and scale
 *
 * @param input_ptr: Input tensor [num_tokens, hidden_size]
 * @param lora_a_ptr: LoRA A weights [num_loras, lora_rank, hidden_size]
 * @param output_ptr: Output tensor [num_slices, num_tokens, lora_rank]
 * @param token_lora_indices: Mapping from token to LoRA ID [num_tokens]
 * @param num_tokens: Number of input tokens
 * @param hidden_size: Hidden dimension size
 * @param lora_rank: LoRA rank dimension
 * @param num_loras: Number of different LoRAs
 * @param num_slices: Number of output slices
 * @param scale: Scaling factor
 * @param input_stride: Stride for input tensor
 * @param lora_stride_0: First stride for LoRA tensor
 * @param lora_stride_1: Second stride for LoRA tensor
 * @param lora_stride_2: Third stride for LoRA tensor
 * @param output_stride_0: First stride for output tensor
 * @param output_stride_1: Second stride for output tensor
 * @param output_stride_2: Third stride for output tensor
 */
void launch_lora_shrink_kernel(
    const void* input_ptr, const void* lora_a_ptr, void* output_ptr,
    const int* token_indices_sorted_ptr, const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr, const int* lora_token_start_loc_ptr,
    int max_active_loras,  // 需要知道活跃 LoRA 的数量
    // --- 原始参数 ---
    int num_total_tokens_in_batch,  // 原来的 num_tokens
    int hidden_size, int lora_rank,
    // int num_loras, (由 max_active_loras 替代)
    int num_slices, float scale, int input_stride, int lora_stride_0,
    int lora_stride_1, int lora_stride_2, int output_stride_0,
    int output_stride_1, int output_stride_2, cudaStream_t stream,
    int input_dtype, int output_dtype);

// Template specializations for different data types
template <typename T>
void lora_shrink_kernel_impl(const T* input, const T* lora_a, T* output,
                             const int* token_lora_indices, int num_tokens,
                             int hidden_size, int lora_rank, int num_loras,
                             int num_slices, float scale, int input_stride,
                             int lora_stride_0, int lora_stride_1,
                             int lora_stride_2, int output_stride_0,
                             int output_stride_1, int output_stride_2,
                             cudaStream_t stream);

extern template void lora_shrink_kernel_impl<half>(const half*, const half*,
                                                   half*, const int*, int, int,
                                                   int, int, int, float, int,
                                                   int, int, int, int, int, int,
                                                   cudaStream_t);

extern template void lora_shrink_kernel_impl<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, const int*, int,
    int, int, int, int, float, int, int, int, int, int, int, int, cudaStream_t);
