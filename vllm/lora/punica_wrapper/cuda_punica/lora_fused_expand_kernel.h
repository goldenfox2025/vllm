#pragma once

#include <cuda_runtime.h>

/**
 * @brief 启动简化版LoRA融合expand kernel
 * 
 * 模仿原始lora_expand_kernel的设计，外部提取好LoRA A数据再传入。
 * 执行标准的 output += lora_a_output @ lora_b_weights
 */
void launch_lora_fused_expand_kernel(
    const void* fused_matmul_output_ptr,        // 融合矩阵乘法后的完整输出
    const void* lora_b_ptr_array,               // slice为索引的LoRA B指针数组
    void* output_ptr,                           // QKV输出张量 [M, qkv_hidden_size]
    const int* token_indices_sorted_ptr,
    const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr,
    const int* lora_token_start_loc_ptr,
    const int* slice_starts_ptr,                // 每个slice在QKV输出中的起始位置
    const int* lora_a_slice_starts_ptr,         // 每个slice在lora_a_output中的起始位置
    const int* lora_slice_ranks_ptr,            // 每个slice的rank
    const int* lora_strides_d0_ptr,
    const int* lora_strides_d1_ptr,
    const int* lora_strides_d2_ptr,
    const int* hidden_sizes_ptr,
    int max_active_loras,
    int num_total_tokens,
    int max_hidden_size,                        // 最大hidden size
    int qkv_output_size,                        // QKV部分的大小，用于偏移
    int num_slices,
    bool add_inputs,
    int fused_input_stride0,                    // 融合输入的stride
    int fused_input_stride1,
    int output_stride0,                         // 输出的stride
    int output_stride1,
    cudaStream_t stream,
    int input_dtype,
    int output_dtype
);



