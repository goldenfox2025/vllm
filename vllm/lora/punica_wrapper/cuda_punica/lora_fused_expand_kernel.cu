#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

/**
 * @brief 简化版QKV+LoRA融合expand CUDA kernel
 * 
 * 模仿原始lora_expand_kernel的设计，外部提取好LoRA A数据再传入
 * 执行标准的 output += lora_a_output @ lora_b_weights
 */
template <typename InputT, typename OutputT>
__global__ void lora_fused_qkv_expand_kernel(
    const InputT* __restrict__ fused_matmul_output,  // 融合矩阵乘法的LoRA A输出部分
    const void* __restrict__ lora_b_ptr_array,       
    OutputT* __restrict__ output,                    
    const int* __restrict__ token_indices_sorted,
    const int* __restrict__ lora_ids,
    const int* __restrict__ num_tokens_per_lora,
    const int* __restrict__ lora_token_start_loc,
    const int* __restrict__ slice_starts,            
    const int* __restrict__ lora_a_slice_starts,     
    const int* __restrict__ lora_slice_ranks,        
    const int* __restrict__ lora_strides_d0,         
    const int* __restrict__ lora_strides_d1,
    const int* __restrict__ lora_strides_d2,
    const int* __restrict__ hidden_sizes,            
    int M,                                           
    int MAX_N,                                       
    int qkv_output_size,                             
    int num_slices,
    int max_active_loras,
    bool add_inputs,
    int fused_input_stride0,                         // 融合输入的stride
    int fused_input_stride1,
    int output_stride0,                              
    int output_stride1) {

    // Grid/Block indexing (same as original Punica kernel)
    int cta_m_num = (M + blockDim.y - 1) / blockDim.y;
    int cta_n_num = (MAX_N + blockDim.x - 1) / blockDim.x;
    int pid_mn = blockIdx.x;
    int pid_m = pid_mn % cta_m_num;
    int pid_n = (pid_mn / cta_m_num) % cta_n_num;
    
    int slice_id = blockIdx.y;
    int lora_idx = blockIdx.z; // Index over active LoRAs
    
    // --- Start of Boundary Checks ---
    if (lora_idx >= max_active_loras || slice_id >= num_slices) {
        return;
    }
    
    int lora_id = lora_ids[lora_idx];
    if (lora_id < 0) { // Inactive LoRA slot
        return;
    }
    
    int num_tokens = num_tokens_per_lora[lora_idx];
    int cta_m_offset = pid_m * blockDim.y;
    if (cta_m_offset >= num_tokens) {
        return;
    }
    
    int token_offset = cta_m_offset + threadIdx.y;
    if (token_offset >= num_tokens) {
        return;
    }
    
    int token_start = lora_token_start_loc[lora_idx];
    int actual_token_idx = token_indices_sorted[token_start + token_offset];
    int hidden_idx = pid_n * blockDim.x + threadIdx.x;
    
    int current_slice_hidden_size = hidden_sizes[slice_id];
    
    if (actual_token_idx < 0) {
        return;
    }
    
    if (hidden_idx >= current_slice_hidden_size || actual_token_idx >= M) {
        return;
    }
    // --- End of Boundary Checks ---

    // --- Main Logic ---
    const int64_t* ptr_values = reinterpret_cast<const int64_t*>(lora_b_ptr_array);
    const InputT* cur_lora_b_ptr = reinterpret_cast<const InputT*>(static_cast<uintptr_t>(ptr_values[slice_id]));
    
    // Get strides for the current slice's LoRA B matrix
    int cur_lora_d0_stride = lora_strides_d0[slice_id];
    int cur_lora_d1_stride = lora_strides_d1[slice_id];
    int cur_lora_d2_stride = lora_strides_d2[slice_id];
    
    // Get rank and LoRA A start position for the current (lora_id, slice_id) pair
    int metadata_idx = lora_id * num_slices + slice_id;
    int slice_rank = lora_slice_ranks[metadata_idx];
    if (slice_rank <= 0) {
        return;
    }
    int lora_a_slice_start = lora_a_slice_starts[metadata_idx];

    // Main computation: accumulator = lora_a_output @ lora_b_weights
    float accumulator = 0.0f;
    for (int k = 0; k < slice_rank; k++) {
        // Correctly calculate the index for lora_a_output
        int col = lora_a_slice_start + k;
        int lora_a_offset = actual_token_idx * fused_input_stride0 + col * fused_input_stride1;
        
        // Correctly calculate the index for lora_b_weights
        int lora_b_offset = lora_id * cur_lora_d0_stride + hidden_idx * cur_lora_d1_stride + k * cur_lora_d2_stride;

        // Perform computation
        float lora_a_val = static_cast<float>(fused_matmul_output[lora_a_offset]);
        float lora_b_val = static_cast<float>(cur_lora_b_ptr[lora_b_offset]);
        
        accumulator += lora_a_val * lora_b_val;
    }
    
    // Final check for accumulator to prevent propagating non-finite values.
    if (!isfinite(accumulator)) {
        return;
    }

    // Write result to the output tensor
    int slice_start = slice_starts[slice_id];
    int output_hidden_idx = slice_start + hidden_idx;
    int output_offset = actual_token_idx * output_stride0 + output_hidden_idx * output_stride1;

    // The total size of the output tensor is M * qkv_output_size
    if (output_offset < 0 || output_offset >= M * qkv_output_size) {
        // This should not happen with correct logic, but as a safeguard.
        return;
    }

    if (add_inputs) {
        output[output_offset] += static_cast<OutputT>(accumulator);
    } else {
        output[output_offset] = static_cast<OutputT>(accumulator);
    }
}

/**
 * @brief 简化版LoRA融合expand kernel实现函数
 */
template <typename InputT, typename OutputT>
void lora_fused_qkv_expand_kernel_impl(
    const InputT* fused_matmul_output, const void* lora_b_ptr_array,
    OutputT* output, const int* token_indices_sorted, const int* lora_ids,
    const int* num_tokens_per_lora, const int* lora_token_start_loc,
    const int* slice_starts, const int* lora_a_slice_starts,
    const int* lora_slice_ranks, const int* lora_strides_d0,
    const int* lora_strides_d1, const int* lora_strides_d2,
    const int* hidden_sizes, int max_active_loras, int M, int MAX_N,
    int qkv_output_size, int num_slices, bool add_inputs, int fused_input_stride0,
    int fused_input_stride1, int output_stride0, int output_stride1,
    cudaStream_t stream) {

    // --- 安全性检查 ---
    if (max_active_loras <= 0 || num_slices <= 0 || M <= 0 || MAX_N <= 0) {
        // printf("Error: Invalid parameters in fused_qkv_expand_kernel\n");
        return;
    }

    // --- 与原始kernel相同的Grid和Block配置 ---
    const int BLOCK_M = 16;
    const int BLOCK_N = 32;

    int cta_m_num = (M + BLOCK_M - 1) / BLOCK_M;
    int cta_n_num = (MAX_N + BLOCK_N - 1) / BLOCK_N;

    // 定义Grid维度，与原始lora_expand_kernel完全一致
    dim3 grid(cta_m_num * cta_n_num, num_slices, max_active_loras);
    dim3 block(BLOCK_N, BLOCK_M);

    // 启动kernel
    lora_fused_qkv_expand_kernel<InputT, OutputT>
        <<<grid, block, 0, stream>>>(
        fused_matmul_output, lora_b_ptr_array, output,
        token_indices_sorted, lora_ids, num_tokens_per_lora, lora_token_start_loc,
        slice_starts, lora_a_slice_starts, lora_slice_ranks,
        lora_strides_d0, lora_strides_d1, lora_strides_d2, hidden_sizes,
        M, MAX_N, qkv_output_size, num_slices, max_active_loras, add_inputs,
        fused_input_stride0, fused_input_stride1, output_stride0, output_stride1);

    // 检查kernel启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA fused expand kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // 等待kernel完成并检查运行时错误
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA fused expand kernel runtime error: %s\n", cudaGetErrorString(err));
    }
}

/**
 * @brief 简化版LoRA融合expand kernel启动函数
 */
void launch_lora_fused_expand_kernel(
    const void* fused_matmul_output_ptr,
    const void* lora_b_ptr_array,
    void* output_ptr,
    const int* token_indices_sorted_ptr,
    const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr,
    const int* lora_token_start_loc_ptr,
    const int* slice_starts_ptr,
    const int* lora_a_slice_starts_ptr,
    const int* lora_slice_ranks_ptr,
    const int* lora_strides_d0_ptr,
    const int* lora_strides_d1_ptr,
    const int* lora_strides_d2_ptr,
    const int* __restrict__ hidden_sizes_ptr,
    int max_active_loras,
    int num_total_tokens,
    int max_hidden_size,
    int qkv_output_size,
    int num_slices,
    bool add_inputs,
    int fused_input_stride0,
    int fused_input_stride1,
    int output_stride0,
    int output_stride1,
    cudaStream_t stream,
    int input_dtype,
    int output_dtype
) {
    // 将输入参数映射到kernel实现函数所需的 M, MAX_N
    int M = num_total_tokens;
    int MAX_N = max_hidden_size;

    // Type dispatch
    if (input_dtype == 1 && output_dtype == 1) { // bf16
        lora_fused_qkv_expand_kernel_impl<__nv_bfloat16, __nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(fused_matmul_output_ptr), lora_b_ptr_array,
            static_cast<__nv_bfloat16*>(output_ptr), token_indices_sorted_ptr, lora_ids_ptr,
            num_tokens_per_lora_ptr, lora_token_start_loc_ptr, slice_starts_ptr,
            lora_a_slice_starts_ptr, lora_slice_ranks_ptr,
            lora_strides_d0_ptr, lora_strides_d1_ptr, lora_strides_d2_ptr,
            hidden_sizes_ptr, max_active_loras, M, MAX_N, qkv_output_size,
            num_slices, add_inputs, fused_input_stride0, fused_input_stride1,
            output_stride0, output_stride1, stream
        );
    } else if (input_dtype == 0 && output_dtype == 0) { // fp16
        lora_fused_qkv_expand_kernel_impl<__half, __half>(
            static_cast<const __half*>(fused_matmul_output_ptr), lora_b_ptr_array,
            static_cast<__half*>(output_ptr), token_indices_sorted_ptr, lora_ids_ptr,
            num_tokens_per_lora_ptr, lora_token_start_loc_ptr, slice_starts_ptr,
            lora_a_slice_starts_ptr, lora_slice_ranks_ptr,
            lora_strides_d0_ptr, lora_strides_d1_ptr, lora_strides_d2_ptr,
            hidden_sizes_ptr, max_active_loras, M, MAX_N, qkv_output_size,
            num_slices, add_inputs, fused_input_stride0, fused_input_stride1,
            output_stride0, output_stride1, stream
        );
    } else if (input_dtype == 2 && output_dtype == 2) { // fp32
        lora_fused_qkv_expand_kernel_impl<float, float>(
            static_cast<const float*>(fused_matmul_output_ptr), lora_b_ptr_array,
            static_cast<float*>(output_ptr), token_indices_sorted_ptr, lora_ids_ptr,
            num_tokens_per_lora_ptr, lora_token_start_loc_ptr, slice_starts_ptr,
            lora_a_slice_starts_ptr, lora_slice_ranks_ptr,
            lora_strides_d0_ptr, lora_strides_d1_ptr, lora_strides_d2_ptr,
            hidden_sizes_ptr, max_active_loras, M, MAX_N, qkv_output_size,
            num_slices, add_inputs, fused_input_stride0, fused_input_stride1,
            output_stride0, output_stride1, stream
        );
    } else {
        printf("Error: Unsupported dtype combination in fused_qkv_expand_kernel. Input: %d, Output: %d\n", 
               input_dtype, output_dtype);
    }
} 

