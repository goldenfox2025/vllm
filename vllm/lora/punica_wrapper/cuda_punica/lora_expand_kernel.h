#ifndef lora_expand_kernel_H
#define lora_expand_kernel_H

#include <cuda_runtime.h>

// 清理版本的expand kernel启动函数声明
void launch_lora_expand_kernel(
    const void* input_ptr, const void* lora_b_ptr, void* output_ptr,
    const int* token_indices_sorted_ptr, const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr, const int* lora_token_start_loc_ptr,
    const int* slice_starts_ptr, const int* lora_strides_d0_ptr,
    const int* lora_strides_d1_ptr, const int* lora_strides_d2_ptr,
    const int* hidden_sizes_ptr, int max_active_loras,
    int num_total_tokens_in_batch, int lora_rank, int hidden_size,
    int num_slices, int offset_start, bool add_inputs, int input_stride0,
    int input_stride1, int input_stride2, int output_stride0,
    int output_stride1, cudaStream_t stream, int input_dtype, int output_dtype);

#endif  // lora_expand_kernel_H
