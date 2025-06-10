/*
 * CUDA LoRA Fused Expand Kernel (CUDA LoRA融合扩展操作核函数)
 *
 * 专门处理QKV+LoRA融合计算的expand操作
 * 输入格式：[num_tokens, total_lora_rank] - 每个token连续存储所有slice的shrink结果
 * 输出格式：[num_tokens, total_hidden_size] - 标准的QKV+LoRA输出
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

/**
 * @brief LoRA Fused Expand CUDA Kernel (LoRA融合扩展操作CUDA核函数)
 *
 * @details
 * 该核函数专门处理QKV+LoRA融合计算后的expand步骤。
 * 融合shrink输入格式：[num_tokens, total_lora_rank]
 * total_lora_rank = max_loras * (slice0_rank + slice1_rank + slice2_rank)
 * 
 * 对于每个token：
 * - token属于特定的LoRA（比如LoRA_i）
 * - 只有LoRA_i对应的部分有数据，其他LoRA部分为0
 * - 需要找到LoRA_i在total_lora_rank中的偏移
 * - 然后在LoRA_i的范围内找到对应slice的偏移
 * 
 * @tparam InputT 输入数据类型
 * @tparam OutputT 输出数据类型
 *
 * @param fused_shrink_input 融合shrink结果 [num_tokens, total_lora_rank]
 * @param lora_b_ptr_array 指向各slice的LoRA B权重张量基地址的指针数组
 * @param output 输出张量 [num_tokens, total_hidden_size]
 * @param token_indices_sorted 为每个LoRA适配器排序后的token索引
 * @param lora_ids 当前批次中活跃的LoRA适配器的ID列表
 * @param num_tokens_per_lora 每个活跃LoRA适配器负责处理的token数量
 * @param lora_token_start_loc 每个活跃LoRA的token起始位置
 * @param slice_starts 每个slice在输出hidden维度上的起始偏移量
 * @param slice_ranks 每个slice的LoRA rank
 * @param hidden_sizes 每个slice的hidden_size
 * @param M 总token数量
 * @param total_hidden_size 总的hidden_size
 * @param num_slices slice数量
 * @param max_loras 最大LoRA数量
 * @param add_inputs 是否累加到输出
 * @param fused_input_stride0 融合输入第0维stride (token维度)
 * @param fused_input_stride1 融合输入第1维stride (total_lora_rank维度)
 * @param output_d0_stride 输出张量第0维stride
 * @param output_d1_stride 输出张量第1维stride
 */
template <typename InputT, typename OutputT>
__global__ void lora_fused_expand_kernel(
    const InputT* fused_shrink_input, const void* lora_b_ptr_array, OutputT* output,
    const int* token_indices_sorted, const int* lora_ids,
    const int* num_tokens_per_lora, const int* lora_token_start_loc,
    const int* slice_starts, const int* slice_ranks,
    const int* lora_strides_d0, const int* lora_strides_d1, const int* lora_strides_d2,
    const int* hidden_sizes, int M, int total_hidden_size, int num_slices, int max_loras,
    bool add_inputs, int fused_input_stride0, int fused_input_stride1,
    int output_d0_stride, int output_d1_stride) {

  // 使用3D grid和3D block
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int slice_id = blockIdx.y;
  int hidden_idx = blockIdx.z * blockDim.z + threadIdx.z;

  // 边界检查
  if (token_idx >= M || slice_id >= num_slices) return;
  
  // 获取当前slice的hidden_size并检查边界
  int current_slice_hidden_size = hidden_sizes[slice_id];
  if (hidden_idx >= current_slice_hidden_size) return;

  // 找到当前token属于哪个LoRA
  int token_lora_idx = -1;
  int token_lora_id = -1;
  
  for (int lora_idx = 0; lora_idx < max_loras; lora_idx++) {
    int lora_id = lora_ids[lora_idx];
    if (lora_id == -1) continue;  // 跳过无效LoRA
    
    int token_start = lora_token_start_loc[lora_idx];
    int num_tokens = num_tokens_per_lora[lora_idx];
    
    // 检查当前token是否在这个LoRA的范围内
    for (int i = 0; i < num_tokens; i++) {
      if (token_indices_sorted[token_start + i] == token_idx) {
        token_lora_idx = lora_idx;
        token_lora_id = lora_id;
        goto found_lora;
      }
    }
  }
  
  found_lora:
  if (token_lora_idx == -1) return;  // 当前token不属于任何LoRA

  // 计算当前LoRA在total_lora_rank中的偏移
  // total_lora_rank = max_loras * (slice0_rank + slice1_rank + slice2_rank)
  int total_rank_per_lora = 0;
  for (int i = 0; i < num_slices; i++) {
    total_rank_per_lora += slice_ranks[i];
  }
  
  int lora_offset_in_total = token_lora_id * total_rank_per_lora;
  
  // 计算当前slice在当前LoRA中的偏移
  int slice_offset_in_lora = 0;
  for (int i = 0; i < slice_id; i++) {
    slice_offset_in_lora += slice_ranks[i];
  }
  
  // 当前slice在total_lora_rank中的起始位置
  int slice_start_in_total = lora_offset_in_total + slice_offset_in_lora;
  int slice_rank = slice_ranks[slice_id];

  // 获取当前slice的LoRA B权重指针
  const int64_t* ptr_values = reinterpret_cast<const int64_t*>(lora_b_ptr_array);
  uintptr_t ptr_value = static_cast<uintptr_t>(ptr_values[slice_id]);
  const __nv_bfloat16* cur_lora_ptr = reinterpret_cast<const __nv_bfloat16*>(ptr_value);

  // 获取当前slice的LoRA B权重矩阵的内存步长
  int cur_lora_d0_stride = lora_strides_d0[slice_id];
  int cur_lora_d1_stride = lora_strides_d1[slice_id];
  int cur_lora_d2_stride = lora_strides_d2[slice_id];

  // 执行矩阵乘法: fused_shrink_input[token, slice_start_in_total:slice_start_in_total+slice_rank] @ lora_b[hidden, rank]
  float accumulator = 0.0f;

  // 沿rank维度进行内积
  for (int k = 0; k < slice_rank; k++) {
    // 计算融合输入的偏移量
    // 融合输入格式：[num_tokens, total_lora_rank]
    int fused_input_offset = token_idx * fused_input_stride0 + 
                            (slice_start_in_total + k) * fused_input_stride1;

    if (fused_input_offset < 0) continue;
    InputT shrink_val = fused_shrink_input[fused_input_offset];

    // 计算LoRA B权重的偏移量
    // 权重矩阵维度: [num_loras, hidden_size_per_slice, rank]
    int weight_offset = token_lora_id * cur_lora_d0_stride +
                        hidden_idx * cur_lora_d1_stride +
                        k * cur_lora_d2_stride;

    if (weight_offset < 0) continue;
    __nv_bfloat16 lora_val = cur_lora_ptr[weight_offset];

    // 类型转换并累加
    float shrink_float = static_cast<float>(shrink_val);
    float lora_float = static_cast<float>(lora_val);
    
    if (isfinite(shrink_float) && isfinite(lora_float)) {
      float product = shrink_float * lora_float;
      if (isfinite(product)) {
        accumulator += product;
      }
    }
  }

  // 计算输出位置
  int slice_start = slice_starts[slice_id];
  int output_hidden_idx = slice_start + hidden_idx;
  int output_offset = token_idx * output_d0_stride + output_hidden_idx * output_d1_stride;

  // 边界检查
  if (output_offset < 0 || output_hidden_idx >= total_hidden_size) return;

  // NaN检查
  if (!isfinite(accumulator)) return;

  // 转换并写入结果
  OutputT result = static_cast<OutputT>(accumulator);

  if (add_inputs) {
    OutputT existing_val = output[output_offset];
    result += existing_val;
  }

  output[output_offset] = result;
}

/**
 * @brief LoRA Fused Expand Kernel 启动函数
 */
template <typename InputT, typename OutputT>
void lora_fused_expand_kernel_impl(
    const InputT* fused_shrink_input, const void* lora_b_ptr_array, OutputT* output,
    const int* token_indices_sorted, const int* lora_ids,
    const int* num_tokens_per_lora, const int* lora_token_start_loc,
    const int* slice_starts, const int* slice_ranks,
    const int* lora_strides_d0, const int* lora_strides_d1, const int* lora_strides_d2,
    const int* hidden_sizes, int max_active_loras, int M, int total_hidden_size, 
    int num_slices, bool add_inputs, int fused_input_stride0, int fused_input_stride1,
    int output_d0_stride, int output_d1_stride, cudaStream_t stream) {

  // 计算最大hidden_size用于grid配置
  int max_hidden_per_slice = total_hidden_size / num_slices;  // 简单估算

  // Block配置
  const int BLOCK_TOKENS = 16;   // token维度的block大小
  const int BLOCK_SLICE = 1;     // slice维度（每个block处理一个slice）
  const int BLOCK_HIDDEN = 32;   // hidden维度的block大小
  
  // Grid配置 - 使用3D grid
  dim3 grid(
    (M + BLOCK_TOKENS - 1) / BLOCK_TOKENS,                          // token维度
    num_slices,                                                      // slice维度  
    (max_hidden_per_slice + BLOCK_HIDDEN - 1) / BLOCK_HIDDEN        // hidden维度
  );
  
  dim3 block(BLOCK_TOKENS, BLOCK_SLICE, BLOCK_HIDDEN);

  // 启动kernel
  lora_fused_expand_kernel<InputT, OutputT><<<grid, block, 0, stream>>>(
      fused_shrink_input, lora_b_ptr_array, output,
      token_indices_sorted, lora_ids, num_tokens_per_lora, lora_token_start_loc,
      slice_starts, slice_ranks, lora_strides_d0, lora_strides_d1, lora_strides_d2, 
      hidden_sizes, M, total_hidden_size, num_slices, max_active_loras, add_inputs,
      fused_input_stride0, fused_input_stride1, output_d0_stride, output_d1_stride);
      
  // 检查kernel启动错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA fused expand kernel launch error: %s\n", cudaGetErrorString(err));
    return;
  }
}

/**
 * @brief LoRA Fused Expand Kernel 外部接口
 */
void launch_lora_fused_expand_kernel(
    const void* fused_shrink_input_ptr, const void* lora_b_ptr_array, void* output_ptr,
    const int* token_indices_sorted_ptr, const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr, const int* lora_token_start_loc_ptr,
    const int* slice_starts_ptr, const int* slice_ranks_ptr, const int* slice_rank_starts_ptr,
    const int* lora_strides_d0_ptr, const int* lora_strides_d1_ptr, const int* lora_strides_d2_ptr,
    const int* hidden_sizes_ptr, int max_active_loras, int num_total_tokens,
    int total_hidden_size, int num_slices, bool add_inputs,
    int fused_input_stride0, int fused_input_stride1, 
    int output_stride0, int output_stride1, cudaStream_t stream,
    int input_dtype, int output_dtype) {

  int M = num_total_tokens;

  // 根据数据类型分发
  if (input_dtype == 2 && output_dtype == 1) {  // float -> bf16
    lora_fused_expand_kernel_impl<float, __nv_bfloat16>(
        static_cast<const float*>(fused_shrink_input_ptr), lora_b_ptr_array,
        static_cast<__nv_bfloat16*>(output_ptr), token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        slice_starts_ptr, slice_ranks_ptr,
        lora_strides_d0_ptr, lora_strides_d1_ptr, lora_strides_d2_ptr,
        hidden_sizes_ptr, max_active_loras, M, total_hidden_size, num_slices, add_inputs,
        fused_input_stride0, fused_input_stride1, output_stride0, output_stride1, stream);
  } else if (input_dtype == 0 && output_dtype == 0) {  // fp16 -> fp16
    lora_fused_expand_kernel_impl<__half, __half>(
        static_cast<const __half*>(fused_shrink_input_ptr), lora_b_ptr_array,
        static_cast<__half*>(output_ptr), token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        slice_starts_ptr, slice_ranks_ptr,
        lora_strides_d0_ptr, lora_strides_d1_ptr, lora_strides_d2_ptr,
        hidden_sizes_ptr, max_active_loras, M, total_hidden_size, num_slices, add_inputs,
        fused_input_stride0, fused_input_stride1, output_stride0, output_stride1, stream);
  } else if (input_dtype == 1 && output_dtype == 1) {  // bf16 -> bf16
    lora_fused_expand_kernel_impl<__nv_bfloat16, __nv_bfloat16>(
        static_cast<const __nv_bfloat16*>(fused_shrink_input_ptr), lora_b_ptr_array,
        static_cast<__nv_bfloat16*>(output_ptr), token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        slice_starts_ptr, slice_ranks_ptr,
        lora_strides_d0_ptr, lora_strides_d1_ptr, lora_strides_d2_ptr,
        hidden_sizes_ptr, max_active_loras, M, total_hidden_size, num_slices, add_inputs,
        fused_input_stride0, fused_input_stride1, output_stride0, output_stride1, stream);
  } else {
    std::cerr << "Fused expand kernel: Unsupported dtype combination: input="
              << input_dtype << ", output=" << output_dtype << std::endl;
  }
} 