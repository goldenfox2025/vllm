/*
 * CUDA LoRA Expand Kernel (CUDA LoRA扩展操作核函数)
 *
 * 执行LoRA扩展操作: output = input @ lora_b_weights (输入乘以LoRA B权重)
 * 支持多个LoRA适配器和slice配置 (例如QKV投影可以看作不同的slice)
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

/**
 * @brief LoRA Expand CUDA Kernel (LoRA扩展操作CUDA核函数)
 *
 * @details
 * 该核函数执行LoRA的扩展步骤，即 `output = input @ lora_b_weights`。
 * - `input` 张量的逻辑维度是 `[num_slices, num_tokens, lora_rank]`。
 * - `lora_b_weights` 张量的逻辑维度是 `[num_loras / num_slices,
 * hidden_size_per_slice, lora_rank]` (具体视外部如何组织lora_b_ptr_array)。
 * - `output` 张量的逻辑维度是 `[num_tokens, total_hidden_size]`。
 *
 * 每一个CUDA线程计算输出矩阵中的一个元素。
 * 线程块被组织起来处理一部分token和一部分hidden_size。
 * `blockIdx.x` 用于二维索引 (pid_m, pid_n) token和hidden维度。
 * `blockIdx.y` 用于索引 slice (例如Q, K, V中的一个)。
 * `blockIdx.z` 用于索引当前处理的LoRA适配器。
 *
 * @tparam InputT 输入数据类型 (例如 __half, __nv_bfloat16, float)
 * @tparam OutputT 输出数据类型 (例如 __half, __nv_bfloat16, float)
 *
 * @param input 输入张量指针 (设备内存)。其形状应理解为 (num_slices, M, K)
 * 通过stride访问。
 * @param lora_b_ptr_array (设备内存中) 指向各slice的LoRA
 * B权重张量基地址的指针数组。数组元素为 uintptr_t (实际指针)。
 * @param output 输出张量指针 (设备内存)。形状为 (M, total_hidden_size)。
 * @param token_indices_sorted (设备内存)
 * 为每个LoRA适配器排序后的token在其原始序列中的索引。
 * @param lora_ids (设备内存) 当前批次中活跃的LoRA适配器的ID列表。
 * @param num_tokens_per_lora (设备内存) 每个活跃LoRA适配器负责处理的token数量。
 * @param lora_token_start_loc (设备内存)
 * 每个活跃LoRA的token在`token_indices_sorted`数组中的起始偏移量。
 * @param slice_starts (设备内存)
 * 每个slice在最终输出张量hidden维度上的起始偏移量。
 * @param lora_strides_d0 (设备内存) LoRA B权重张量在第0维(通常对应LoRA
 * ID或adapter)的步长。注意：此参数在当前kernel实现中未被用于寻址权重，可能暗示权重指针`cur_lora_ptr`已指向特定LoRA
 * adapter的权重，或者权重组织方式不依赖此stride。
 * @param lora_strides_d1 (设备内存) LoRA
 * B权重张量在第1维(对应hidden_size)的步长。
 * @param lora_strides_d2 (设备内存) LoRA B权重张量在第2维(对应lora_rank
 * K)的步长。
 * @param hidden_sizes (设备内存) 每个slice的hidden_size。
 * @param M 当前批处理中的总token数 (num_total_tokens_in_batch)。
 * @param MAX_N 单个slice的最大hidden_size
 * (用于grid计算，确保覆盖所有可能的hidden_idx)。
 * @param K LoRA的秩 (rank)。
 * @param num_slices slice的数量 (例如，QKV分别是一个slice，则num_slices=3)。
 * @param add_inputs 布尔值，指示是否将计算结果累加到`output`张量的现有值上。
 * @param input_d0_stride 输入张量第0维(slice)的步长。
 * @param input_d1_stride 输入张量第1维(token)的步长。
 * @param input_d2_stride 输入张量第2维(K, lora_rank)的步长。
 * @param output_d0_stride 输出张量第0维(token)的步长。
 * @param output_d1_stride 输出张量第1维(hidden_size)的步长。
 */
template <typename InputT, typename OutputT>
__global__ void lora_expand_kernel(
    const InputT* input, const void* lora_b_ptr_array, OutputT* output,
    const int* token_indices_sorted, const int* lora_ids,
    const int* num_tokens_per_lora, const int* lora_token_start_loc,
    const int* slice_starts, const int* lora_strides_d0,
    const int* lora_strides_d1, const int* lora_strides_d2,
    const int* hidden_sizes, int M, int MAX_N, int K, int num_slices,
    bool add_inputs, int input_d0_stride, int input_d1_stride,
    int input_d2_stride, int output_d0_stride, int output_d1_stride) {

  // Triton: pid_mn = tl.program_id(axis=0)
  //         pid_m = pid_mn % cta_m_num
  //         pid_n = (pid_mn // cta_m_num) % cta_n_num
  int cta_m_num = (M + blockDim.y - 1) / blockDim.y;
  int cta_n_num = (MAX_N + blockDim.x - 1) / blockDim.x;
  int pid_mn = blockIdx.x;
  int pid_m = pid_mn % cta_m_num;
  int pid_n = (pid_mn / cta_m_num) % cta_n_num;
  // slice_id: 当前处理的slice (例如Q, K, 或 V)
  int slice_id = blockIdx.y;
  // lora_idx: 当前处理的LoRA适配器在活跃列表中的索引 (从0开始)
  int lora_idx = blockIdx.z;  // blockIdx.z 范围是 [0, max_active_loras - 1]

  // tid_m: token方向的线程在块内的索引
  int tid_m = threadIdx.y;
  // tid_n: hidden_size方向的线程在块内的索引
  int tid_n = threadIdx.x;

  // 边界检查：确保slice_id和lora_idx有效
  if (slice_id >= num_slices) return;

  // 获取当前LoRA适配器的ID
  int lora_id = lora_ids[lora_idx];
  // 如果LoRA ID为-1，表示该槽位未使用或无效，则不处理
  if (lora_id == -1) return;


  // Triton: cta_m_offset = pid_m * BLOCK_M
  //         token_offset = cta_m_offset + tid_m
  int token_start = lora_token_start_loc[lora_idx];
  int num_tokens = num_tokens_per_lora[lora_idx];
  int cta_m_offset = pid_m * blockDim.y;  // BLOCK_M

  // 边界检查：确保cta_m_offset在当前LoRA的token数量范围内
  if (cta_m_offset >= num_tokens) return;

  int token_offset = cta_m_offset + tid_m;
  // 边界检查：确保token_offset在当前LoRA的token数量范围内
  if (token_offset >= num_tokens) return;

  // 获取实际的token索引
  int actual_token_idx = token_indices_sorted[token_start + token_offset];

  // 计算hidden索引，与Triton匹配
  // Triton: offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
  int hidden_idx = pid_n * blockDim.x + tid_n;  // BLOCK_N


  int current_slice_hidden_size = hidden_sizes[slice_id];
  if (hidden_idx >= current_slice_hidden_size || actual_token_idx >= M) {
    return;
  }

  // 获取当前slice的LoRA B权重指针
  // lora_b_ptr_array包含指向每个slice权重基地址的指针 (以int64_t形式存储)
  const int64_t* ptr_values =
      reinterpret_cast<const int64_t*>(lora_b_ptr_array);
  // 根据slice_id获取对应slice的权重基地址 (uintptr_t)
  uintptr_t ptr_value = static_cast<uintptr_t>(ptr_values[slice_id]);

  const __nv_bfloat16* cur_lora_ptr =
      reinterpret_cast<const __nv_bfloat16*>(ptr_value);

  // 获取当前slice的LoRA B权重矩阵的内存步长
  int cur_lora_d0_stride = lora_strides_d0[slice_id];  // LoRA ID维度的stride
  int cur_lora_d1_stride = lora_strides_d1[slice_id];  // hidden维度的stride
  int cur_lora_d2_stride = lora_strides_d2[slice_id];  // K维度的stride

  // 执行矩阵乘法: input[token, k] * weight[hidden, k]
  // (对于当前lora_id和slice_id) weight的维度是 [hidden_size_per_slice, K]
  // input的维度是 [K] (对应 specific token, specific slice)
  float accumulator = 0.0f;  // 使用float进行累加，保证精度

  // 主计算循环 (沿K维度进行内积)
  for (int k = 0; k < K; k++) {
    // 计算输入张量的偏移量
    // input 张量维度: [num_slices, num_tokens, lora_rank]
    int input_offset = slice_id * input_d0_stride +
                       actual_token_idx * input_d1_stride + k * input_d2_stride;

    if (input_offset < 0 || k >= K) continue;
    InputT input_val = input[input_offset];

    // 计算LoRA B权重张量的偏移量
    // 权重矩阵维度: [num_loras, hidden_size_per_slice, K]
    // 修复：根据Triton实现，应该使用lora_index（即lora_id）来索引权重
    // Triton版本使用: cur_lora_d0_stride * lora_index
    int weight_offset = lora_id * cur_lora_d0_stride +
                        hidden_idx * cur_lora_d1_stride +
                        k * cur_lora_d2_stride;

    if (weight_offset < 0) continue;
    __nv_bfloat16 lora_val = cur_lora_ptr[weight_offset];

    float input_float = static_cast<float>(input_val);
    float lora_float = static_cast<float>(lora_val);
    if (!isfinite(input_float) || !isfinite(lora_float)) continue;

    // 累加结果
    float product = input_float * lora_float;
    if (isfinite(product)) {
      accumulator += product;
    }
  }

  // 计算输出位置
  int slice_start =
      slice_starts[slice_id];  // 当前slice在总hidden_size中的起始位置
  // 输出的hidden索引是slice的起始位置加上当前slice内的hidden_idx
  int output_hidden_idx = slice_start + hidden_idx;
  // 计算输出张量的偏移量
  // output 张量维度: [num_tokens, total_hidden_size]
  int output_offset = actual_token_idx * output_d0_stride +
                      output_hidden_idx * output_d1_stride;

  // output_d0_stride是第0维的stride，不是hidden_size的总大小
  // 应该检查output_hidden_idx是否超出实际的hidden_size范围
  int total_hidden_size =
      output_d0_stride;  // 对于2D输出张量，stride[0]就是hidden_size
  if (output_offset < 0 || actual_token_idx >= M || output_hidden_idx < 0 ||
      output_hidden_idx >= total_hidden_size) {
    return;
  }

  // 添加NaN检查，防止NaN传播
  if (!isfinite(accumulator)) {
    return;  // 如果累加器包含NaN或Inf，直接返回
  }

  // 转换累加结果为输出类型
  OutputT result = static_cast<OutputT>(accumulator);

  if (add_inputs) {
    OutputT existing_val = output[output_offset];
    result += existing_val;
  }

  // 写入输出
  output[output_offset] = result;
}

template <typename... Args>
bool is_contiguous(Args... args) {
  int strides[] = {args...};
  for (int i = 0; i < sizeof(strides) / sizeof(strides[0]) - 1; ++i) {
    if (strides[i] < strides[i + 1]) {
      return false;
    }
  }
  return true;
}

/**
 * @brief LoRA Expand Kernel 实现函数 (LoRA扩展操作Kernel的模板实现与启动封装)
 *
 * @details
 * 此函数负责设置CUDA核函数 `lora_expand_kernel` 的执行配置
 * (grid维度、block维度) 并启动它。
 *
 * @tparam InputT 输入数据类型
 * @tparam OutputT 输出数据类型
 *
 * @param input 输入张量指针 (设备内存)
 * @param lora_b_ptr_array 指向LoRA B权重张量指针的数组 (设备内存)
 * @param output 输出张量指针 (设备内存)
 * @param token_indices_sorted 每个LoRA中已排序的token索引 (设备内存)
 * @param lora_ids 当前批次中活跃的LoRA ID数组 (设备内存)
 * @param num_tokens_per_lora 每个LoRA对应的token数量 (设备内存)
 * @param lora_token_start_loc 每个LoRA在token_indices_sorted中的起始位置
 * (设备内存)
 * @param slice_starts 每个slice在输出张量hidden维度上的起始位置 (设备内存)
 * @param lora_strides_d0 LoRA B权重张量的第0维stride (设备内存)
 * @param lora_strides_d1 LoRA B权重张量的第1维stride (设备内存)
 * @param lora_strides_d2 LoRA B权重张量的第2维stride (设备内存)
 * @param hidden_sizes 每个slice的hidden_size大小 (设备内存)
 * @param max_active_loras 当前批次中最大的活跃LoRA适配器数量
 * (用于设置gridDim.z)
 * @param M 总token数量 (num_total_tokens_in_batch)
 * @param MAX_N 单个slice的最大hidden_size (用于grid计算)
 * @param K LoRA的秩 (rank)
 * @param num_slices 分片数量
 * @param add_inputs 是否将计算结果累加到输出张量中
 * @param input_d0_stride 输入张量的第0维stride
 * @param input_d1_stride 输入张量的第1维stride
 * @param input_d2_stride 输入张量的第2维stride
 * @param output_d0_stride 输出张量的第0维stride
 * @param output_d1_stride 输出张量的第1维stride
 * @param stream CUDA流
 */
template <typename InputT, typename OutputT>
void lora_expand_kernel_impl(
    const InputT* input, const void* lora_b_ptr_array, OutputT* output,
    const int* token_indices_sorted, const int* lora_ids,
    const int* num_tokens_per_lora, const int* lora_token_start_loc,
    const int* slice_starts, const int* lora_strides_d0,
    const int* lora_strides_d1, const int* lora_strides_d2,
    const int* hidden_sizes, int max_active_loras, int M, int MAX_N, int K,
    int num_slices, bool add_inputs, int input_d0_stride, int input_d1_stride,
    int input_d2_stride, int output_d0_stride, int output_d1_stride,
    cudaStream_t stream) {
  // 修复：Grid 和 Block 配置，使用GPU兼容的配置
  // Triton使用BLOCK_M=64, BLOCK_N=128，但CUDA需要考虑线程数限制
  // 最大线程数通常是1024，所以使用较小的block配置
  const int BLOCK_M = 16;  // 每个block在M维(token)处理的元素数量
  const int BLOCK_N = 32;  // 每个block在N维(hidden_size)处理的元素数量

  // 修复：计算grid维度，与Triton完全匹配
  // Triton grid: (triton.cdiv(M, BLOCK_M) * triton.cdiv(MAX_N, BLOCK_N),
  // NUM_SLICES, MAX_LORAS)
  int cta_m_num = (M + BLOCK_M - 1) / BLOCK_M;
  int cta_n_num = (MAX_N + BLOCK_N - 1) / BLOCK_N;

  // 定义Grid维度，与Triton完全一致
  // grid.x: 处理 (token, hidden_size) 平面的块数量
  // grid.y: 处理 slice 的数量 (每个slice一个平面)
  // grid.z: 处理活跃LoRA适配器的数量 (每个LoRA一个深度)
  dim3 grid(cta_m_num * cta_n_num, num_slices, max_active_loras);
  // 定义Block维度
  // block.x: N维 (hidden_size方向) 的线程数
  // block.y: M维 (token方向) 的线程数
  dim3 block(BLOCK_N, BLOCK_M);  // blockDim.x=BLOCK_N, blockDim.y=BLOCK_M

  // 启动kernel
  lora_expand_kernel<InputT, OutputT><<<grid, block, 0, stream>>>(
      input, lora_b_ptr_array, output, token_indices_sorted, lora_ids,
      num_tokens_per_lora, lora_token_start_loc, slice_starts, lora_strides_d0,
      lora_strides_d1, lora_strides_d2, hidden_sizes, M, MAX_N, K, num_slices,
      add_inputs, input_d0_stride, input_d1_stride, input_d2_stride,
      output_d0_stride, output_d1_stride);

  // 检查kernel启动是否有错误
  // (异步错误需要后续的cudaStreamSynchronize或类似操作才能捕获运行时错误)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "CUDA clean expand kernel launch error: %s\n",  // "clean expand kernel"
                                                        // 可能是项目或模块名
        cudaGetErrorString(err));
  }
}

/**
 * @brief LoRA Expand Kernel 主启动函数 (LoRA扩展操作的外部调用接口)
 *
 * @details
 * 此函数根据输入 (`input_dtype`) 和输出 (`output_dtype`) 的数据类型，
 * 选择并调用特定模板实例化的 `lora_expand_kernel_impl` 函数。
 *
 * @param input_ptr 输入张量指针 (void*, 指向设备内存)
 * @param lora_b_ptr 指向LoRA B权重张量指针的数组 (void*, 设备内存中的指针数组,
 * 每个元素是一个uintptr_t)
 * @param output_ptr 输出张量指针 (void*, 指向设备内存)
 * @param token_indices_sorted_ptr (设备内存)
 * 为每个LoRA适配器排序后的token在其原始序列中的索引
 * @param lora_ids_ptr (设备内存) 当前批次中活跃的LoRA适配器的ID列表
 * @param num_tokens_per_lora_ptr (设备内存)
 * 每个活跃LoRA适配器负责处理的token数量
 * @param lora_token_start_loc_ptr (设备内存)
 * 每个活跃LoRA的token在`token_indices_sorted`数组中的起始偏移量
 * @param slice_starts_ptr (设备内存)
 * 每个slice在最终输出张量hidden维度上的起始偏移量
 * @param lora_strides_d0_ptr (设备内存) LoRA B权重张量在第0维的步长
 * @param lora_strides_d1_ptr (设备内存) LoRA B权重张量在第1维的步长
 * @param lora_strides_d2_ptr (设备内存) LoRA B权重张量在第2维的步长
 * @param hidden_sizes_ptr (设备内存) 每个slice的hidden_size
 * @param max_active_loras 当前批次中最大的活跃LoRA适配器数量
 * @param num_total_tokens_in_batch 当前批次中的总token数量 (对应kernel中的M)
 * @param lora_rank LoRA的秩 (对应kernel中的K)
 * @param hidden_size 单个slice的最大hidden_size (对应kernel中的MAX_N)
 * @param num_slices 分片数量
 * @param offset_start (未使用) 输出的偏移起始位置 (此参数在当前代码中未被使用)
 * @param add_inputs 是否将计算结果累加到输出张量中
 * @param input_stride0 输入张量第0维(slice)的步长
 * @param input_stride1 输入张量第1维(token)的步长
 * @param input_stride2 输入张量第2维(K, lora_rank)的步长
 * @param output_stride0 输出张量第0维(token)的步长
 * @param output_stride1 输出张量第1维(hidden_size)的步长
 * @param stream CUDA流
 * @param input_dtype 输入数据类型枚举值 (例如: 0 for fp16, 1 for bf16, 2 for
 * float)
 * @param output_dtype 输出数据类型枚举值 (例如: 0 for fp16, 1 for bf16)
 */
void launch_lora_expand_kernel(
    const void* input_ptr, const void* lora_b_ptr, void* output_ptr,
    const int* token_indices_sorted_ptr, const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr, const int* lora_token_start_loc_ptr,
    const int* slice_starts_ptr, const int* lora_strides_d0_ptr,
    const int* lora_strides_d1_ptr, const int* lora_strides_d2_ptr,
    const int* hidden_sizes_ptr, int max_active_loras,
    int num_total_tokens_in_batch, int lora_rank,
    int hidden_size,  // hidden_size这里用作MAX_N
    int num_slices, int offset_start, bool add_inputs, int input_stride0,
    int input_stride1, int input_stride2, int output_stride0,
    int output_stride1, cudaStream_t stream, int input_dtype,
    int output_dtype) {
  // 将输入参数映射到kernel实现函数所需的 M, MAX_N, K
  int M = num_total_tokens_in_batch;
  int MAX_N = hidden_size;  // 此处的hidden_size参数被用作MAX_N
  int K = lora_rank;

  // 根据输入和输出的数据类型进行分发
  if (input_dtype == 2 && output_dtype == 1) {  // 输入: float, 输出: bf16
    lora_expand_kernel_impl<float, __nv_bfloat16>(
        static_cast<const float*>(input_ptr), lora_b_ptr,
        static_cast<__nv_bfloat16*>(output_ptr), token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        slice_starts_ptr, lora_strides_d0_ptr, lora_strides_d1_ptr,
        lora_strides_d2_ptr, hidden_sizes_ptr, max_active_loras, M, MAX_N, K,
        num_slices, add_inputs, input_stride0, input_stride1, input_stride2,
        output_stride0, output_stride1, stream);
  } else if (input_dtype == 0 && output_dtype == 0) {  // 输入: fp16, 输出: fp16
    lora_expand_kernel_impl<__half, __half>(
        static_cast<const __half*>(input_ptr), lora_b_ptr,
        static_cast<__half*>(output_ptr), token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        slice_starts_ptr, lora_strides_d0_ptr, lora_strides_d1_ptr,
        lora_strides_d2_ptr, hidden_sizes_ptr, max_active_loras, M, MAX_N, K,
        num_slices, add_inputs, input_stride0, input_stride1, input_stride2,
        output_stride0, output_stride1, stream);
  } else if (input_dtype == 1 && output_dtype == 1) {  // 输入: bf16, 输出: bf16
    lora_expand_kernel_impl<__nv_bfloat16, __nv_bfloat16>(
        static_cast<const __nv_bfloat16*>(input_ptr), lora_b_ptr,
        static_cast<__nv_bfloat16*>(output_ptr), token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        slice_starts_ptr, lora_strides_d0_ptr, lora_strides_d1_ptr,
        lora_strides_d2_ptr, hidden_sizes_ptr, max_active_loras, M, MAX_N, K,
        num_slices, add_inputs, input_stride0, input_stride1, input_stride2,
        output_stride0, output_stride1, stream);
  } else {
    // 如果有其他数据类型组合的需求，可以在这里添加更多的else if分支
    std::cerr << "Clean kernel: Unsupported dtype combination: input="
              << input_dtype << ", output=" << output_dtype << std::endl;
  }
}