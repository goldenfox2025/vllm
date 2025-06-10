#ifndef LORA_FUSED_EXPAND_KERNEL_H
#define LORA_FUSED_EXPAND_KERNEL_H

#include <cuda_runtime.h>

/**
 * @brief Launch LoRA Fused Expand Kernel
 * 
 * 专门处理QKV+LoRA融合计算的expand操作
 * 输入格式：[num_tokens, total_lora_rank] - 每个token连续存储所有slice的shrink结果
 * total_lora_rank = max_loras * (slice0_rank + slice1_rank + slice2_rank)
 */
void launch_lora_fused_expand_kernel(
    const void* fused_shrink_input_ptr,      // 融合shrink结果 [num_tokens, total_lora_rank]
    const void* lora_b_ptr_array,            // LoRA B权重指针数组
    void* output_ptr,                        // 输出 [num_tokens, total_hidden_size]
    const int* token_indices_sorted_ptr,     // 排序的token索引
    const int* lora_ids_ptr,                 // 活跃LoRA ID列表
    const int* num_tokens_per_lora_ptr,      // 每个LoRA的token数量
    const int* lora_token_start_loc_ptr,     // 每个LoRA的token起始位置
    const int* slice_starts_ptr,             // 每个slice在输出中的起始位置
    const int* slice_ranks_ptr,              // 每个slice的rank
    const int* slice_rank_starts_ptr,        // 保留参数（未使用，保持接口兼容性）
    const int* lora_strides_d0_ptr,          // LoRA B权重stride
    const int* lora_strides_d1_ptr,
    const int* lora_strides_d2_ptr,
    const int* hidden_sizes_ptr,             // 每个slice的hidden_size
    int max_active_loras,                    // 最大活跃LoRA数量
    int num_total_tokens,                    // 总token数量
    int total_hidden_size,                   // 总hidden_size
    int num_slices,                          // slice数量
    bool add_inputs,                         // 是否累加输入
    int fused_input_stride0,                 // 融合输入的stride
    int fused_input_stride1,
    int output_stride0,                      // 输出的stride
    int output_stride1,
    cudaStream_t stream,                     // CUDA流
    int input_dtype,                         // 输入数据类型
    int output_dtype                         // 输出数据类型
);

#endif // LORA_FUSED_EXPAND_KERNEL_H 