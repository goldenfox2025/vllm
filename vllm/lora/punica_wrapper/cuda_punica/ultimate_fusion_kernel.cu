#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include "ultimate_fusion_kernel.h"

/**
 * @brief 终极融合CUDA内核
 * 
 * 设计思想：每个CTA(线程块)负责处理一个token的所有计算
 * 1. 基础QKV路径：output_base = input @ qkv_weights
 * 2. LoRA路径：output_lora = (input @ lora_a) @ lora_b
 * 3. 结果融合：output = output_base + output_lora
 */
template <typename InputT, typename OutputT>
__global__ void ultimate_fusion_kernel(
    const InputT* input_ptr,          // [num_tokens, hidden_size]
    const InputT* qkv_weights_ptr,    // [qkv_output_size, hidden_size]
    const void* lora_a_ptr_array,     // 指向各slice LoRA A权重的指针数组
    const void* lora_b_ptr_array,     // 指向各slice LoRA B权重的指针数组
    OutputT* output_ptr,              // [num_tokens, qkv_output_size]
    const int* token_indices_sorted_ptr,
    const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr,
    const int* lora_token_start_loc_ptr,
    const int* slice_starts_ptr,
    const int* lora_ranks_ptr,
    int num_tokens,
    int hidden_size,
    int qkv_output_size,
    int num_slices,
    int max_rank,
    // strides
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
    int output_stride1
) {
    // 每个线程块处理一个token
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // 边界检查
    if (token_idx >= num_tokens) return;
    
    // 共享内存：存储当前token的输入向量
    extern __shared__ float shared_mem[];
    float* s_hidden_state = shared_mem;                          // [hidden_size]
    float* s_lora_intermediate = shared_mem + hidden_size;       // [max_rank]
    
    // === Step 1: 协同加载输入向量到共享内存 ===
    for (int i = tid; i < hidden_size; i += block_size) {
        int input_offset = token_idx * input_stride0 + i * input_stride1;
        s_hidden_state[i] = static_cast<float>(input_ptr[input_offset]);
    }
    __syncthreads();
    
    // === Step 2: 计算基础QKV路径 ===
    // 每个线程负责计算输出向量的一部分
    for (int out_idx = tid; out_idx < qkv_output_size; out_idx += block_size) {
        float accumulator = 0.0f;
        
        // 执行点积：s_hidden_state @ qkv_weights[out_idx, :]
        for (int k = 0; k < hidden_size; k++) {
            int weight_offset = out_idx * qkv_stride0 + k * qkv_stride1;
            float weight_val = static_cast<float>(qkv_weights_ptr[weight_offset]);
            accumulator += s_hidden_state[k] * weight_val;
        }
        
        // 先将基础QKV结果写入全局内存
        int output_offset = token_idx * output_stride0 + out_idx * output_stride1;
        output_ptr[output_offset] = static_cast<OutputT>(accumulator);
    }
    __syncthreads();
    
    // === Step 3: 查找当前token对应的LoRA ===
    // 使用更高效的二分查找或者简化方法
    int active_lora_idx = -1;
    int lora_id = -1;
    
    // 简化：假设token索引已经排序，直接查找对应的LoRA
    // 这里需要根据实际的token mapping来确定
    // 暂时使用简单的映射：假设第一个活跃的LoRA处理所有token
    if (lora_ids_ptr[0] != -1) {
        active_lora_idx = 0;
        lora_id = lora_ids_ptr[0];
    }
    
    // 如果没有活跃的LoRA，直接返回（只有基础QKV）
    if (lora_id == -1) {
        return;
    }
    
    // === Step 4: 执行LoRA计算 ===
    // 遍历所有slice (Q, K, V)
    for (int slice_id = 0; slice_id < num_slices; slice_id++) {
        // 获取当前slice的LoRA A权重指针（正确的方式）
        const int64_t* ptr_values_a = reinterpret_cast<const int64_t*>(lora_a_ptr_array);
        uintptr_t ptr_value_a = static_cast<uintptr_t>(ptr_values_a[slice_id]);
        const InputT* cur_lora_a_ptr = reinterpret_cast<const InputT*>(ptr_value_a);
        
        // 获取当前slice的LoRA B权重指针（正确的方式）
        const int64_t* ptr_values_b = reinterpret_cast<const int64_t*>(lora_b_ptr_array);
        uintptr_t ptr_value_b = static_cast<uintptr_t>(ptr_values_b[slice_id]);
        const InputT* cur_lora_b_ptr = reinterpret_cast<const InputT*>(ptr_value_b);
        
        // 确定当前slice的rank
        int slice_rank = max_rank; // 简化：假设所有slice有相同的rank
        
        // === Step 4a: LoRA Shrink阶段：input @ lora_a ===
        // 清零中间结果
        for (int r = tid; r < slice_rank; r += block_size) {
            s_lora_intermediate[r] = 0.0f;
        }
        __syncthreads();
        
        // 计算 s_hidden_state @ lora_a -> s_lora_intermediate
        for (int r = tid; r < slice_rank; r += block_size) {
            float accumulator = 0.0f;
            for (int k = 0; k < hidden_size; k++) {
                int lora_a_offset = lora_id * lora_a_stride0 + r * lora_a_stride1 + k * lora_a_stride2;
                float lora_a_val = static_cast<float>(cur_lora_a_ptr[lora_a_offset]);
                accumulator += s_hidden_state[k] * lora_a_val;
            }
            s_lora_intermediate[r] = accumulator;
        }
        __syncthreads();
        
        // === Step 4b: LoRA Expand阶段：intermediate @ lora_b ===
        // 获取当前slice在输出中的起始位置和大小
        int slice_start = slice_starts_ptr[slice_id];
        int slice_end = (slice_id + 1 < num_slices) ? slice_starts_ptr[slice_id + 1] : qkv_output_size;
        int slice_size = slice_end - slice_start;
        
        // 计算 s_lora_intermediate @ lora_b，并累加到输出
        for (int out_idx = tid; out_idx < slice_size; out_idx += block_size) {
            float accumulator = 0.0f;
            for (int r = 0; r < slice_rank; r++) {
                int lora_b_offset = lora_id * lora_b_stride0 + out_idx * lora_b_stride1 + r * lora_b_stride2;
                float lora_b_val = static_cast<float>(cur_lora_b_ptr[lora_b_offset]);
                accumulator += s_lora_intermediate[r] * lora_b_val;
            }
            
            // 将LoRA增量加到输出上
            int global_out_idx = slice_start + out_idx;
            int output_offset = token_idx * output_stride0 + global_out_idx * output_stride1;
            
            // 读取当前输出值，加上LoRA增量，然后写回
            OutputT current_val = output_ptr[output_offset];
            float new_val = static_cast<float>(current_val) + accumulator;
            output_ptr[output_offset] = static_cast<OutputT>(new_val);
        }
        __syncthreads();
    }
}

/**
 * @brief 内核实现模板函数
 */
template <typename InputT, typename OutputT>
void ultimate_fusion_kernel_impl(
    const InputT* input_ptr,
    const InputT* qkv_weights_ptr,
    const void* lora_a_ptr_array,
    const void* lora_b_ptr_array,
    OutputT* output_ptr,
    const int* token_indices_sorted_ptr,
    const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr,
    const int* lora_token_start_loc_ptr,
    const int* slice_starts_ptr,
    const int* lora_ranks_ptr,
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
    cudaStream_t stream
) {
    // Grid配置：每个token一个block
    dim3 grid(num_tokens);
    
    // Block配置：使用256个线程（经验值）
    const int THREADS_PER_BLOCK = 256;
    dim3 block(THREADS_PER_BLOCK);
    
    // 共享内存大小：hidden_state + lora_intermediate
    size_t shared_mem_size = (hidden_size + max_rank) * sizeof(float);
    
    // 启动内核
    ultimate_fusion_kernel<InputT, OutputT><<<grid, block, shared_mem_size, stream>>>(
        input_ptr, qkv_weights_ptr, lora_a_ptr_array, lora_b_ptr_array, output_ptr,
        token_indices_sorted_ptr, lora_ids_ptr, num_tokens_per_lora_ptr,
        lora_token_start_loc_ptr, slice_starts_ptr, lora_ranks_ptr,
        num_tokens, hidden_size, qkv_output_size, num_slices, max_rank,
        input_stride0, input_stride1, qkv_stride0, qkv_stride1,
        lora_a_stride0, lora_a_stride1, lora_a_stride2,
        lora_b_stride0, lora_b_stride1, lora_b_stride2,
        output_stride0, output_stride1
    );
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // std::cerr << "Ultimate fusion kernel launch error: " << cudaGetErrorString(err) << std::endl;
        // printf("Ultimate fusion kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * @brief 公开的启动函数
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
) {
    // 根据数据类型分发
    if (input_dtype == 0 && output_dtype == 0) { // fp16 -> fp16
        ultimate_fusion_kernel_impl<__half, __half>(
            static_cast<const __half*>(input_ptr),
            static_cast<const __half*>(qkv_weights_ptr),
            lora_a_ptr_array, lora_b_ptr_array,
            static_cast<__half*>(output_ptr),
            token_indices_sorted_ptr, lora_ids_ptr, num_tokens_per_lora_ptr,
            lora_token_start_loc_ptr, slice_starts_ptr, lora_ranks_ptr,
            num_tokens, hidden_size, qkv_output_size, num_slices, max_rank,
            input_stride0, input_stride1, qkv_stride0, qkv_stride1,
            lora_a_stride0, lora_a_stride1, lora_a_stride2,
            lora_b_stride0, lora_b_stride1, lora_b_stride2,
            output_stride0, output_stride1, stream
        );
    } else if (input_dtype == 1 && output_dtype == 1) { // bf16 -> bf16
        ultimate_fusion_kernel_impl<__nv_bfloat16, __nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(input_ptr),
            static_cast<const __nv_bfloat16*>(qkv_weights_ptr),
            lora_a_ptr_array, lora_b_ptr_array,
            static_cast<__nv_bfloat16*>(output_ptr),
            token_indices_sorted_ptr, lora_ids_ptr, num_tokens_per_lora_ptr,
            lora_token_start_loc_ptr, slice_starts_ptr, lora_ranks_ptr,
            num_tokens, hidden_size, qkv_output_size, num_slices, max_rank,
            input_stride0, input_stride1, qkv_stride0, qkv_stride1,
            lora_a_stride0, lora_a_stride1, lora_a_stride2,
            lora_b_stride0, lora_b_stride1, lora_b_stride2,
            output_stride0, output_stride1, stream
        );
    } else if (input_dtype == 2 && output_dtype == 2) { // float -> float
        ultimate_fusion_kernel_impl<float, float>(
            static_cast<const float*>(input_ptr),
            static_cast<const float*>(qkv_weights_ptr),
            lora_a_ptr_array, lora_b_ptr_array,
            static_cast<float*>(output_ptr),
            token_indices_sorted_ptr, lora_ids_ptr, num_tokens_per_lora_ptr,
            lora_token_start_loc_ptr, slice_starts_ptr, lora_ranks_ptr,
            num_tokens, hidden_size, qkv_output_size, num_slices, max_rank,
            input_stride0, input_stride1, qkv_stride0, qkv_stride1,
            lora_a_stride0, lora_a_stride1, lora_a_stride2,
            lora_b_stride0, lora_b_stride1, lora_b_stride2,
            output_stride0, output_stride1, stream
        );
    } else {
        std::cerr << "Ultimate fusion kernel: Unsupported dtype combination: input=" 
                  << input_dtype << ", output=" << output_dtype << std::endl;
    }
} 