#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <iostream>

constexpr int BLOCK_M = 32;  // 与 Triton 一致
constexpr int BLOCK_N = 16;  // 与 Triton 一致
constexpr int BLOCK_K = 32;

template <typename InputT, typename OutputT>
__global__ void lora_shrink_kernel_triton_style(
    const InputT* __restrict__ input,   // [num_tokens, hidden_size]
    const InputT* __restrict__ lora_a,  // [num_loras, lora_rank, hidden_size]
    OutputT* __restrict__ output,       // [num_slices, num_tokens, lora_rank]
    const int* __restrict__ token_indices_sorted,  // [num_tokens] - 按 LoRA ID
                                                   // 排序的 token 索引
    const int* __restrict__ lora_ids,  // [max_loras] - LoRA ID 列表
    const int* __restrict__ num_tokens_per_lora,  // [max_loras] - 每个 LoRA 的
                                                  // token 数量
    const int* __restrict__ lora_token_start_loc,  // [max_loras+1] - 每个 LoRA
                                                   // 在 token_indices_sorted
                                                   // 中的起始位置
    int M,           // num_tokens
    int N,           // lora_rank
    int K,           // hidden_size
    int num_slices,  // 输出切片数
    float scaling,   // 缩放因子
    // strides
    int input_d0_stride,   // input stride[0]
    int input_d1_stride,   // input stride[1]
    int lora_d0_stride,    // lora stride[0] (lora_id)
    int lora_d1_stride,    // lora stride[1] (rank)
    int lora_d2_stride,    // lora stride[2] (hidden)
    int output_d0_stride,  // output stride[0] (slice)
    int output_d1_stride,  // output stride[1] (token)
    int output_d2_stride   // output stride[2] (rank)
) {
  // 计算MN方向上总共需要多少个线程块
  int cta_n_num = (N + BLOCK_N - 1) / BLOCK_N;
  int cta_m_num = (M + BLOCK_M - 1) / BLOCK_M;

  // 当前线程块处理的数据
  int cta_m_idx = blockIdx.x % cta_m_num;
  int cta_n_idx = blockIdx.x / cta_m_num % cta_n_num;

  /*
  MergedQKVParallelLinearWithLoRA是vllm内部的合并层
  多层合并可以提高效率 mark
  而slice id 在这里决定对应qkv的哪一个
 */

  int slice_id = blockIdx.y;

  // 使用哪一个lora
  int lora_idx = blockIdx.z;

  // 检查是否使用 LoRA
  int lora_id = lora_ids[lora_idx];
  if (lora_id == -1) {
    return;
  }

  int lora_m_size = num_tokens_per_lora[lora_idx];

  // 当前线程块负责的部分 在M方向的偏移
  int cta_m_offset = cta_m_idx * BLOCK_M;
  if (cta_m_offset >= lora_m_size) {
    return;
  }

  // 当前线程块负责的部分 具体长度
  int cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset);

  // 当前loraidx决定的线程块 负责的token的偏移量
  int lora_m_indices_start = lora_token_start_loc[lora_idx];

  // 朴素分组
  int tid_m = threadIdx.y;
  int tid_n = threadIdx.x;

  // 当前线程块负责的部分 在N方向的偏移
  int cta_n_offset = cta_n_idx * BLOCK_N;
  if (cta_n_offset >= N) return;

  int cta_n_len = min(BLOCK_N, N - cta_n_offset);

  // 提前检查索引是否超线程块长度限制
  bool thread_active = (tid_m < cta_m_len && tid_n < cta_n_len);

  // 累加器初始化
  float accumulator = 0.0f;

  // 先不使用split-k
  for (int k_offset = 0; k_offset < K; k_offset += BLOCK_K) {
    int k_len = min(BLOCK_K, K - k_offset);

    if (thread_active) {
      // 获取当前线程对应的 token 索引
      int token_idx_in_lora = cta_m_offset + tid_m;
      if (token_idx_in_lora < lora_m_size) {
        // 从 token_indices_sorted 中获取实际的 token 索引
        int actual_token_idx =
            token_indices_sorted[lora_m_indices_start + token_idx_in_lora];
        // 计算输出的 rank 索引
        int rank_idx = cta_n_offset + tid_n;

        // 执行点积计算
        // sum(input[token, k] * lora_a[lora_id, rank, k])
        for (int k = 0; k < k_len; k++) {
          int k_global = k_offset + k;
          if (k_global < K) {
            // 获取输入值: input[actual_token_idx, k_global]
            InputT input_val = input[actual_token_idx * input_d0_stride +
                                     k_global * input_d1_stride];

            InputT lora_val =
                lora_a[slice_id * lora_d0_stride + rank_idx * lora_d1_stride +
                       k_global * lora_d2_stride];

            // 累加
            accumulator +=
                static_cast<float>(input_val) * static_cast<float>(lora_val);
          }
        }
      }
    }
  }

  if (thread_active) {
    int token_idx_in_lora = cta_m_offset + tid_m;
    if (token_idx_in_lora < lora_m_size) {
      // 从 token_indices_sorted 中获取实际的 token 索引
      int actual_token_idx =
          token_indices_sorted[lora_m_indices_start + token_idx_in_lora];
      // 计算输出的 rank 索引
      int rank_idx = cta_n_offset + tid_n;

      // 应用缩放并写入输出
      OutputT result = static_cast<OutputT>(accumulator * scaling);

      // output[slice_id, actual_token_idx, rank_idx]
      int output_offset = slice_id * output_d0_stride +
                          actual_token_idx * output_d1_stride +
                          rank_idx * output_d2_stride;

      output[output_offset] = result;
    }
  }
}

template <typename InputT, typename OutputT>
void lora_shrink_kernel_impl(
    const InputT* input, const InputT* lora_a, OutputT* output,
    const int* token_indices_sorted, const int* lora_ids,
    const int* num_tokens_per_lora, const int* lora_token_start_loc,
    int max_active_loras, int M, int N, int K, int num_slices, float scaling,
    int input_d0_stride, int input_d1_stride, int lora_d0_stride,
    int lora_d1_stride, int lora_d2_stride, int output_d0_stride,
    int output_d1_stride, int output_d2_stride, cudaStream_t stream) {
  // 复刻 Triton 的 grid 计算
  int cta_m_num = (M + BLOCK_M - 1) / BLOCK_M;
  int cta_n_num = (N + BLOCK_N - 1) / BLOCK_N;

  // Grid: (cta_m_num * cta_n_num, num_slices, max_active_loras)
  dim3 grid(cta_m_num * cta_n_num, num_slices, max_active_loras + 1);

  // Block: (BLOCK_N, BLOCK_M) - 与 Triton 的 block 大小一致
  dim3 block(BLOCK_N, BLOCK_M);

  lora_shrink_kernel_triton_style<InputT, OutputT><<<grid, block, 0, stream>>>(
      input, lora_a, output, token_indices_sorted, lora_ids,
      num_tokens_per_lora, lora_token_start_loc, M, N, K, num_slices, scaling,
      input_d0_stride, input_d1_stride, lora_d0_stride, lora_d1_stride,
      lora_d2_stride, output_d0_stride, output_d1_stride, output_d2_stride);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error (Triton-style): %s\n",
           cudaGetErrorString(err));
  }
}

void launch_lora_shrink_kernel(
    const void* input_ptr, const void* lora_a_ptr, void* output_ptr,
    const int* token_indices_sorted_ptr, const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr, const int* lora_token_start_loc_ptr,
    int max_active_loras, int num_total_tokens_in_batch, int hidden_size,
    int lora_rank, int num_slices, float scale, int input_stride,
    int lora_stride_0, int lora_stride_1, int lora_stride_2,
    int output_stride_0, int output_stride_1, int output_stride_2,
    cudaStream_t stream, int input_dtype, int output_dtype) {
  int M = num_total_tokens_in_batch;  // num_tokens
  int N = lora_rank;                  // lora_rank
  int K = hidden_size;                // hidden_size

  if (input_dtype == 0 && output_dtype == 2) {  // half -> float
    lora_shrink_kernel_impl<half, float>(
        static_cast<const half*>(input_ptr),
        static_cast<const half*>(lora_a_ptr), static_cast<float*>(output_ptr),
        token_indices_sorted_ptr, lora_ids_ptr, num_tokens_per_lora_ptr,
        lora_token_start_loc_ptr, max_active_loras, M, N, K, num_slices, scale,
        input_stride, 1,                                    // input strides
        lora_stride_0, lora_stride_1, lora_stride_2,        // lora strides
        output_stride_0, output_stride_1, output_stride_2,  // output strides
        stream);
  } else if (input_dtype == 0 && output_dtype == 0) {  // half -> half
    lora_shrink_kernel_impl<half, half>(
        static_cast<const half*>(input_ptr),
        static_cast<const half*>(lora_a_ptr), static_cast<half*>(output_ptr),
        token_indices_sorted_ptr, lora_ids_ptr, num_tokens_per_lora_ptr,
        lora_token_start_loc_ptr, max_active_loras, M, N, K, num_slices, scale,
        input_stride, 1,                                    // input strides
        lora_stride_0, lora_stride_1, lora_stride_2,        // lora strides
        output_stride_0, output_stride_1, output_stride_2,  // output strides
        stream);
  } else if (input_dtype == 1 && output_dtype == 2) {  // bf16 -> float
    lora_shrink_kernel_impl<__nv_bfloat16, float>(
        static_cast<const __nv_bfloat16*>(input_ptr),
        static_cast<const __nv_bfloat16*>(lora_a_ptr),
        static_cast<float*>(output_ptr), token_indices_sorted_ptr, lora_ids_ptr,
        num_tokens_per_lora_ptr, lora_token_start_loc_ptr, max_active_loras, M,
        N, K, num_slices, scale, input_stride, 1,           // input strides
        lora_stride_0, lora_stride_1, lora_stride_2,        // lora strides
        output_stride_0, output_stride_1, output_stride_2,  // output strides
        stream);
  } else {
    std::cerr << "Unsupported dtype combination: input=" << input_dtype
              << ", output=" << output_dtype << std::endl;
  }
}