#ifndef ULTIMATE_FUSION_KERNEL_H
#define ULTIMATE_FUSION_KERNEL_H

#include <cuda_runtime.h>

/**
 * @brief 终极融合内核启动函数
 * 
 * 在一个内核中同时完成：
 * 1. 基础QKV线性变换: output_base = input @ qkv_weights  
 * 2. LoRA路径计算: output_lora = (input @ lora_a) @ lora_b
 * 3. 结果融合: output = output_base + output_lora (仅对使用LoRA的token)
 * 
 * @param input_ptr 输入张量 [num_tokens, hidden_size]
 * @param qkv_weights_ptr QKV权重矩阵 [qkv_output_size, hidden_size]
 * @param lora_a_ptr_array 指向各slice LoRA A权重的指针数组
 * @param lora_b_ptr_array 指向各slice LoRA B权重的指针数组  
 * @param output_ptr 输出张量 [num_tokens, qkv_output_size]
 * @param token_indices_sorted_ptr Punica排序的token索引
 * @param lora_ids_ptr 活跃的LoRA ID数组
 * @param num_tokens_per_lora_ptr 每个LoRA的token数量
 * @param lora_token_start_loc_ptr 每个LoRA在排序数组中的起始位置
 * @param slice_starts_ptr 每个slice在输出中的起始位置
 * @param lora_ranks_ptr 每个LoRA的rank数组
 * @param max_active_loras 最大活跃LoRA数量
 * @param num_tokens 总token数量
 * @param hidden_size 隐藏层大小
 * @param qkv_output_size QKV输出大小
 * @param num_slices slice数量
 * @param max_rank 最大LoRA rank
 * @param input_stride0 输入张量stride[0]
 * @param input_stride1 输入张量stride[1] 
 * @param qkv_stride0 QKV权重stride[0]
 * @param qkv_stride1 QKV权重stride[1]
 * @param lora_a_stride0 LoRA A权重stride[0] (lora_id维度)
 * @param lora_a_stride1 LoRA A权重stride[1] (rank维度)
 * @param lora_a_stride2 LoRA A权重stride[2] (hidden维度)
 * @param lora_b_stride0 LoRA B权重stride[0] (lora_id维度)
 * @param lora_b_stride1 LoRA B权重stride[1] (output维度) 
 * @param lora_b_stride2 LoRA B权重stride[2] (rank维度)
 * @param output_stride0 输出张量stride[0]
 * @param output_stride1 输出张量stride[1]
 * @param stream CUDA流
 * @param input_dtype 输入数据类型 (0=fp16, 1=bf16, 2=fp32)
 * @param output_dtype 输出数据类型
 */
void launch_ultimate_fusion_kernel(
    const void* input_ptr,
    const void* qkv_weights_ptr, 
    const void* lora_a_ptr_array,
    const void* lora_b_ptr_array,
    void* output_ptr,
    const int* token_indices_sorted_ptr,
    const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr,
    const int* lora_token_start_loc_ptr,
    const int* slice_starts_ptr,
    const int* lora_ranks_ptr,
    int max_active_loras,
    int num_tokens,
    int hidden_size,
    int qkv_output_size,
    int num_slices,
    int max_rank,
    int input_stride0,
    int input_stride1,
    int qkv_stride0,
    int qkv_stride1,
    int lora_a_stride0,
    int lora_a_stride1, 
    int lora_a_stride2,
    int lora_b_stride0,
    int lora_b_stride1,
    int lora_b_stride2,
    int output_stride0,
    int output_stride1,
    cudaStream_t stream,
    int input_dtype,
    int output_dtype
);

#endif // ULTIMATE_FUSION_KERNEL_H 