#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <mma.h>    // 引入 WMMA 头文件
#include <string>   // 用于std::string
#include <cstdlib>  // 用于std::getenv

template <typename InputT, typename OutputT, int BLOCK_M = 32, int BLOCK_N = 32,
          int BLOCK_K = 32>
__global__ void lora_shrink_kernel_v0(
    const InputT* __restrict__ input,           // [num_tokens, hidden_size]
    const void* __restrict__ lora_a_ptr_array,  // 指向每个slice权重的指针数组
    OutputT* __restrict__ output,  // [num_slices, num_tokens, lora_rank]
    const int* __restrict__ token_indices_sorted,  // [num_tokens] - 按 LoRA ID
                                                   // 排序的 token 索引
    const int* __restrict__ lora_ids,              // [max_loras] - LoRA ID 列表
    const int* __restrict__ num_tokens_per_lora,   // [max_loras] - 每个 LoRA 的
                                                   // token 数量
    const int* __restrict__ lora_token_start_loc,  // [max_loras+1] - 每个 LoRA
                                                   // 在 token_indices_sorted
                                                   // 中的起始位置
    int M,                                         // num_tokens
    int N,                                         // lora_rank
    int K,                                         // hidden_size
    int num_slices,                                // 输出切片数
    float scaling,                                 // 缩放因子
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

  int SPLIT_K = 1;
  int pid_sk_m_n = blockIdx.x;
  int pid_sk = pid_sk_m_n % SPLIT_K;
  int cta_m_idx = (pid_sk_m_n / SPLIT_K) % cta_m_num;
  int cta_n_idx = pid_sk_m_n / (SPLIT_K * cta_m_num) % cta_n_num;

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

  const InputT* cur_lora_ptr;
  if (num_slices == 1) {
    // 单个slice情况：直接使用权重指针
    cur_lora_ptr = reinterpret_cast<const InputT*>(lora_a_ptr_array);
  } else {
    // 多个slice情况：从指针数组中获取
    const int64_t* ptr_values =
        reinterpret_cast<const int64_t*>(lora_a_ptr_array);
    uintptr_t ptr_value = static_cast<uintptr_t>(ptr_values[slice_id]);
    cur_lora_ptr = reinterpret_cast<const InputT*>(ptr_value);
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

            // Triton版本使用: cur_lora_ptr + lora_d0_stride * lora_index + ...
            // cur_lora_ptr已经指向当前slice的权重，权重布局: [num_loras,
            // lora_rank, hidden_size]
            InputT lora_val = cur_lora_ptr[lora_id * lora_d0_stride +
                                           rank_idx * lora_d1_stride +
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

template <typename InputT, typename OutputT, int BLOCK_M = 32, int BLOCK_N = 32,
          int BLOCK_K = 32>
void lora_shrink_kernel_impl_v0(
    const InputT* input, const void* lora_a_ptr_array, OutputT* output,
    const int* token_indices_sorted, const int* lora_ids,
    const int* num_tokens_per_lora, const int* lora_token_start_loc,
    int max_active_loras, int M, int N, int K, int num_slices, float scaling,
    int input_d0_stride, int input_d1_stride, int lora_d0_stride,
    int lora_d1_stride, int lora_d2_stride, int output_d0_stride,
    int output_d1_stride, int output_d2_stride, cudaStream_t stream) {
  int cta_m_num = (M + BLOCK_M - 1) / BLOCK_M;
  int cta_n_num = (N + BLOCK_N - 1) / BLOCK_N;

  int SPLIT_K = 1;
  dim3 grid(SPLIT_K * cta_m_num * cta_n_num, num_slices, max_active_loras);

  dim3 block(BLOCK_N, BLOCK_M);

  lora_shrink_kernel_v0<InputT, OutputT, BLOCK_M, BLOCK_N, BLOCK_K>
      <<<grid, block, 0, stream>>>(
          input, lora_a_ptr_array, output, token_indices_sorted, lora_ids,
          num_tokens_per_lora, lora_token_start_loc, M, N, K, num_slices,
          scaling, input_d0_stride, input_d1_stride, lora_d0_stride,
          lora_d1_stride, lora_d2_stride, output_d0_stride, output_d1_stride,
          output_d2_stride);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error (Triton-style): %s\n",
           cudaGetErrorString(err));
  }
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

template <typename InputT, typename OutputT, int TM, int TN, int BLOCK_M = 32,
          int BLOCK_N = 32, int BLOCK_K = 32>
__global__ void lora_shrink_kernel_v1(
    const InputT* __restrict__ input,           // [num_tokens, hidden_size]
    const void* __restrict__ lora_a_ptr_array,  // 指向每个slice权重的指针数组
    OutputT* __restrict__ output,  // [num_slices, num_tokens, lora_rank]
    const int* __restrict__ token_indices_sorted,  // [num_tokens] - 按 LoRA ID
                                                   // 排序的 token 索引
    const int* __restrict__ lora_ids,              // [max_loras] - LoRA ID 列表
    const int* __restrict__ num_tokens_per_lora,   // [max_loras] - 每个 LoRA 的
                                                   // token 数量
    const int* __restrict__ lora_token_start_loc,  // [max_loras+1] - 每个 LoRA
                                                   // 在 token_indices_sorted
                                                   // 中的起始位置
    int M,                                         // num_tokens
    int N,                                         // lora_rank
    int K,                                         // hidden_size
    int num_slices,                                // 输出切片数
    float scaling,                                 // 缩放因子
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

  int SPLIT_K = 1;
  int pid_sk_m_n = blockIdx.x;
  int pid_sk = pid_sk_m_n % SPLIT_K;
  int cta_m_idx = (pid_sk_m_n / SPLIT_K) % cta_m_num;
  int cta_n_idx = pid_sk_m_n / (SPLIT_K * cta_m_num) % cta_n_num;

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

  const InputT* cur_lora_ptr;
  if (num_slices == 1) {
    // 单个slice情况：直接使用权重指针
    cur_lora_ptr = reinterpret_cast<const InputT*>(lora_a_ptr_array);
  } else {
    // 多个slice情况：从指针数组中获取
    const int64_t* ptr_values =
        reinterpret_cast<const int64_t*>(lora_a_ptr_array);
    uintptr_t ptr_value = static_cast<uintptr_t>(ptr_values[slice_id]);
    cur_lora_ptr = reinterpret_cast<const InputT*>(ptr_value);
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
  int tid = threadIdx.x;

  // 当前线程块负责的部分 在N方向的偏移
  int cta_n_offset = cta_n_idx * BLOCK_N;
  if (cta_n_offset >= N) return;

  int cta_n_len = min(BLOCK_N, N - cta_n_offset);

  // 一个线程块负责BLOCK_M*BLOCK_N
  __shared__ float smem_input[BLOCK_M][BLOCK_K + 1];
  __shared__ float smem_lora[BLOCK_N][BLOCK_K + 1];

  constexpr int TILES_IN_M = BLOCK_M / TM;
  constexpr int TILES_IN_N = BLOCK_N / TN;

  // 计算二维的 tile 编号
  int tile_m_idx = tid % TILES_IN_M;
  int tile_n_idx = tid / TILES_IN_M;

  // 计算在 shared memory 中的起始坐标
  int tid_m = tile_m_idx * TM;
  int tid_n = tile_n_idx * TN;
  // 一个线程负责4*4的结果
  float accumulator[TM][TN];
  for (int m = 0; m < TM; ++m) {
    for (int n = 0; n < TN; ++n) {
      accumulator[m][n] = 0.0f;
    }
  }

  for (int k_offset = 0; k_offset < K; k_offset += BLOCK_K) {
    // 加载input
    for (int load_idx = tid; load_idx < BLOCK_M * BLOCK_K;
         load_idx += blockDim.x) {
      int load_m = load_idx / BLOCK_K;
      int load_k = load_idx % BLOCK_K;
      int global_load_m = cta_m_offset + load_m;
      int global_load_k = k_offset + load_k;

      if (global_load_m < lora_m_size && global_load_k < K) {
        int actual_token_idx =
            token_indices_sorted[lora_m_indices_start + global_load_m];
        smem_input[load_m][load_k] =
            static_cast<float>(input[actual_token_idx * input_d0_stride +
                                     global_load_k * input_d1_stride]);
      } else {
        smem_input[load_m][load_k] = 0.0f;
      }
    }
    // 加载lora
    for (int load_idx = tid; load_idx < BLOCK_N * BLOCK_K;
         load_idx += blockDim.x) {
      int load_n = load_idx / BLOCK_K;
      int load_k = load_idx % BLOCK_K;
      // 边界检查
      int rank_idx = cta_n_offset + load_n;
      int k_global = k_offset + load_k;
      if (rank_idx < N && k_global < K) {
        smem_lora[load_n][load_k] = static_cast<float>(
            cur_lora_ptr[lora_id * lora_d0_stride + rank_idx * lora_d1_stride +
                         k_global * lora_d2_stride]);
      } else {
        smem_lora[load_n][load_k] = 0.0f;
      }
    }

    __syncthreads();

    for (int m = 0; m < TM; ++m) {
      for (int n = 0; n < TN; ++n) {
        if ((tid_m + m) < BLOCK_M && (tid_n + n) < BLOCK_N) {
          for (int k = 0; k < BLOCK_K; ++k) {
            accumulator[m][n] +=
                smem_input[tid_m + m][k] * smem_lora[tid_n + n][k];
          }
        }
      }
    }
    __syncthreads();
  }

  // 写回HBM
  for (int m = 0; m < TM; ++m) {
    for (int n = 0; n < TN; ++n) {
      int token_idx_in_lora = cta_m_offset + tid_m + m;
      int rank_idx = cta_n_offset + tid_n + n;
      if (token_idx_in_lora < lora_m_size && rank_idx < N) {
        int actual_token_idx =
            token_indices_sorted[lora_m_indices_start + token_idx_in_lora];
        int output_offset = slice_id * output_d0_stride +
                            actual_token_idx * output_d1_stride +
                            rank_idx * output_d2_stride;
        output[output_offset] =
            static_cast<OutputT>(accumulator[m][n] * scaling);
      }
    }
  }
}
template <typename InputT, typename OutputT, int BLOCK_M = 32, int BLOCK_N = 32,
          int BLOCK_K = 32>
void lora_shrink_kernel_impl_v1(
    const InputT* input, const void* lora_a_ptr_array, OutputT* output,
    const int* token_indices_sorted, const int* lora_ids,
    const int* num_tokens_per_lora, const int* lora_token_start_loc,
    int max_active_loras, int M, int N, int K, int num_slices, float scaling,
    int input_d0_stride, int input_d1_stride, int lora_d0_stride,
    int lora_d1_stride, int lora_d2_stride, int output_d0_stride,
    int output_d1_stride, int output_d2_stride, cudaStream_t stream) {
  int cta_m_num = (M + BLOCK_M - 1) / BLOCK_M;
  int cta_n_num = (N + BLOCK_N - 1) / BLOCK_N;

  // 规模太小的时候，TM和TN不能大，否则warp会太少。
  constexpr int TM = 1;
  constexpr int TN = 1;
  int SPLIT_K = 1;
  dim3 grid(SPLIT_K * cta_m_num * cta_n_num, num_slices, max_active_loras);
  constexpr int THREAD_NUM = BLOCK_M * BLOCK_N / (TM * TN);
  dim3 block(THREAD_NUM);
  if (!is_contiguous(lora_d0_stride, lora_d1_stride, lora_d2_stride) ||
      !is_contiguous(output_d0_stride, output_d1_stride, output_d2_stride) ||
      !is_contiguous(input_d0_stride, input_d1_stride)) {
    std::cerr << "lora_d0_stride: " << lora_d0_stride << std::endl;
    std::cerr << "lora_d1_stride: " << lora_d1_stride << std::endl;
    std::cerr << "lora_d2_stride: " << lora_d2_stride << std::endl;
    std::cerr << "output_d0_stride: " << output_d0_stride << std::endl;
    std::cerr << "output_d1_stride: " << output_d1_stride << std::endl;
    std::cerr << "output_d2_stride: " << output_d2_stride << std::endl;
    std::cerr << "input_d0_stride: " << input_d0_stride << std::endl;
    std::cerr << "input_d1_stride: " << input_d1_stride << std::endl;
    throw std::runtime_error(
        "lora_shrink_kernel_impl_v1: lora_d0_stride, lora_d1_stride, "
        "lora_d2_stride, output_d0_stride, output_d1_stride, output_d2_stride, "
        "input_d0_stride, input_d1_stride must be contiguous");
  }
  lora_shrink_kernel_v1<InputT, OutputT, TM, TN, BLOCK_M, BLOCK_N, BLOCK_K>
      <<<grid, block, 0, stream>>>(
          input, lora_a_ptr_array, output, token_indices_sorted, lora_ids,
          num_tokens_per_lora, lora_token_start_loc, M, N, K, num_slices,
          scaling, input_d0_stride, input_d1_stride, lora_d0_stride,
          lora_d1_stride, lora_d2_stride, output_d0_stride, output_d1_stride,
          output_d2_stride);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error (Triton-style): %s\n",
           cudaGetErrorString(err));
  }
}

// 定义一个基于输入数据类型的向量化读取内核
template <typename InputT, int N>
union Vec {
  InputT data[N];
  float4 f4;
};

/* v2 版：⽀持 Split‑K
- grid.x = SPLIT_K × cta_m_num × cta_n_num
- 每个 SPLIT_K block 对 [K/SPLIT_K] 区间做乘加，计算完毕后通过 atomicAdd
- 写回同⼀ output 位置，实现跨 block 的 K 维归并。
- 当 SPLIT_K == 1 时，与 v1 完全等价，⽆额外开销。 */

template <typename InputT, typename OutputT, int TM, int TN, int BLOCK_M = 32,
          int BLOCK_N = 32, int BLOCK_K = 32>
__global__ void lora_shrink_kernel_v2(
    const InputT* __restrict__ input,           // [num_tokens, hidden_size]
    const void* __restrict__ lora_a_ptr_array,  // slice -> weight ptr
    OutputT* __restrict__ output,  // [num_slices, num_tokens, lora_rank]
    const int* __restrict__ token_indices_sorted,
    const int* __restrict__ lora_ids,
    const int* __restrict__ num_tokens_per_lora,
    const int* __restrict__ lora_token_start_loc, int M, int N, int K,
    int num_slices, float scaling,
    // strides
    int input_d0_stride, int input_d1_stride, int lora_d0_stride,
    int lora_d1_stride, int lora_d2_stride, int output_d0_stride,
    int output_d1_stride, int output_d2_stride) {
  // 确定当前线程块负责的mn维度的总块数
  const int cta_n_num = (N + BLOCK_N - 1) / BLOCK_N;
  const int cta_m_num = (M + BLOCK_M - 1) / BLOCK_M;
  const int SPLIT_K = gridDim.x / (cta_m_num * cta_n_num);  // 运行时可变

  const int pid_sk_m_n = blockIdx.x;
  // 按照split k分解
  const int pid_sk = pid_sk_m_n % SPLIT_K;
  // 除以split k后， 按照m维度分解
  const int cta_m_idx = (pid_sk_m_n / SPLIT_K) % cta_m_num;
  // 除以split k和m维度后，按照n维度分解
  const int cta_n_idx = pid_sk_m_n / (SPLIT_K * cta_m_num) % cta_n_num;

  const int slice_id = blockIdx.y;
  const int lora_idx = blockIdx.z;

  const int lora_id = lora_ids[lora_idx];
  if (lora_id == -1) return;  // 未启⽤ LoRA

  /* 获取当前 slice 的 A 权重指针 */
  const InputT* cur_lora_ptr;
  if (num_slices == 1) {
    cur_lora_ptr = reinterpret_cast<const InputT*>(lora_a_ptr_array);
  } else {
    const int64_t* ptr_values =
        reinterpret_cast<const int64_t*>(lora_a_ptr_array);
    uintptr_t ptr_value = static_cast<uintptr_t>(ptr_values[slice_id]);
    cur_lora_ptr = reinterpret_cast<const InputT*>(ptr_value);
  }

  // 得到当前lora需要处理的token长度
  const int lora_m_size = num_tokens_per_lora[lora_idx];
  // 当前线程块负责的token的起始位置
  const int cta_m_offset = cta_m_idx * BLOCK_M;
  // 起始位置大于token长度，直接返回
  if (cta_m_offset >= lora_m_size) return;

  // 当前线程块负责的token长度
  const int cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset);
  const int lora_m_idx_start = lora_token_start_loc[lora_idx];

  // 当前线程块负责的rank维度 也就是N维度
  const int cta_n_offset = cta_n_idx * BLOCK_N;
  if (cta_n_offset >= N) return;

  const int cta_n_len = min(BLOCK_N, N - cta_n_offset);

  // 当前线程块负责的K维度
  const int k_per_split = (K + SPLIT_K - 1) / SPLIT_K;
  const int k_begin = pid_sk * k_per_split;
  const int k_end = min(K, k_begin + k_per_split);

  // 共享内存
  __shared__ float smem_input[BLOCK_M][BLOCK_K + 1];
  __shared__ float smem_lora[BLOCK_N][BLOCK_K + 1];

  constexpr int TILES_IN_M = BLOCK_M / TM;
  constexpr int TILES_IN_N = BLOCK_N / TN;

  const int tid = threadIdx.x;
  const int tile_n_id = tid % TILES_IN_N;
  const int tile_m_id = tid / TILES_IN_N;
  const int tid_m = tile_m_id * TM;
  const int tid_n = tile_n_id * TN;

  float accumulator[TM][TN];
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) accumulator[m][n] = 0.f;
  // 尝试向量化读写
  constexpr int VEC_UNIT = sizeof(float4) / sizeof(InputT);
  Vec<InputT, VEC_UNIT> vec_;
  // 主循环：遍历 K
  for (int k_off = k_begin; k_off < k_end; k_off += BLOCK_K) {
    for (int idx = tid * VEC_UNIT; idx < BLOCK_M * BLOCK_K;
         idx += blockDim.x * VEC_UNIT) {
      int lm = idx / BLOCK_K;
      int lk = idx % BLOCK_K;
      int gm = cta_m_offset + lm;
      int gk = k_off + lk;
      if (gm >= lora_m_size) {
        // 这个线程负责的行已经超出了当前LoRA的处理范围，直接填充0即可
        for (int i = 0; i < VEC_UNIT; ++i) {
          if (lk + i < BLOCK_K) {
            smem_input[lm][lk + i] = 0.f;
          }
        }
        continue;  // 继续处理下一个线程负责的加载任务
      }
      // 获取经过排序和映射后的真实 token_idx，并定位到输入数据的行首地址
      const int actual_token_idx = token_indices_sorted[lora_m_idx_start + gm];
      const InputT* input_row = &input[actual_token_idx * input_d0_stride];
      if (gk + VEC_UNIT <= k_end) {
        using VecT = Vec<InputT, VEC_UNIT>;
        float4 f4_val = reinterpret_cast<const float4*>(&input_row[gk])[0];
        VecT* vec_val = reinterpret_cast<VecT*>(&f4_val);

        for (int i = 0; i < VEC_UNIT; ++i) {
          smem_input[lm][lk + i] = static_cast<float>(vec_val->data[i]);
        }
      } else {
        for (int i = 0; i < VEC_UNIT; ++i) {
          if (gk + i < k_end) {
            smem_input[lm][lk + i] = static_cast<float>(input_row[gk + i]);
          } else {
            // 超出K边界的部分填充0
            smem_input[lm][lk + i] = 0.f;
          }
        }
      }
    }

    // 载入 LoRA A[N,K] 到 shared
    for (int idx = tid; idx < BLOCK_N * BLOCK_K; idx += blockDim.x) {
      int ln = idx / BLOCK_K;
      int lk = idx % BLOCK_K;
      int gr = cta_n_offset + ln;  // rank idx
      int gk = k_off + lk;
      smem_lora[ln][lk] =
          (gr < N && gk < k_end)
              ? static_cast<float>(
                    cur_lora_ptr[lora_id * lora_d0_stride +
                                 gr * lora_d1_stride + gk * lora_d2_stride])
              : 0.f;
    }
    __syncthreads();

    // 片上 GEMM 累加
#pragma unroll
    for (int m = 0; m < TM; ++m)
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        if ((tid_m + m) < BLOCK_M && (tid_n + n) < BLOCK_N) {
#pragma unroll
          for (int k = 0; k < BLOCK_K; ++k)
            accumulator[m][n] +=
                smem_input[tid_m + m][k] * smem_lora[tid_n + n][k];
        }
      }
    __syncthreads();
  } /* for k_off */

// 写回（带 Split‑K 归并）
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      const int tok_in_lora = cta_m_offset + tid_m + m;
      const int rank_idx = cta_n_offset + tid_n + n;
      if (tok_in_lora < lora_m_size && rank_idx < N) {
        const int actual_tok =
            token_indices_sorted[lora_m_idx_start + tok_in_lora];
        const int out_off = slice_id * output_d0_stride +
                            actual_tok * output_d1_stride +
                            rank_idx * output_d2_stride;

        const float val = accumulator[m][n] * scaling;

        // Split‑K 归并
        // 这里全局内存访问不合并
        // n方向上，线程是连续的。所以outoff理论上也连续 应该可以合并

        if (SPLIT_K == 1) {
          output[out_off] = static_cast<OutputT>(val);
        } else {
          if constexpr (std::is_same_v<OutputT, float>) {
            atomicAdd(&output[out_off], val);
          } else if constexpr (std::is_same_v<OutputT, half>) {
            atomicAdd(&output[out_off], static_cast<half>(val));
          } else if constexpr (std::is_same_v<OutputT, __nv_bfloat16>) {
            atomicAdd(&output[out_off], static_cast<__nv_bfloat16>(val));
          }
        }
      }
    }
}

/* v2 impl：调度 Split‑K 版本的 kernel
 * - 参数列表保持不变
 * - 根据 K 和实际硬件线程上限⾃动推导合适的 SPLIT_K */
template <typename InputT, typename OutputT, int BLOCK_M = 32, int BLOCK_N = 32,
          int BLOCK_K = 32>
void lora_shrink_kernel_impl_v2(
    const InputT* input, const void* lora_a_ptr_array, OutputT* output,
    const int* token_indices_sorted, const int* lora_ids,
    const int* num_tokens_per_lora, const int* lora_token_start_loc,
    int max_active_loras, int M, int N, int K, int num_slices, float scaling,
    int input_d0_stride, int input_d1_stride, int lora_d0_stride,
    int lora_d1_stride, int lora_d2_stride, int output_d0_stride,
    int output_d1_stride, int output_d2_stride, cudaStream_t stream) {
  const int cta_m_num = (M + BLOCK_M - 1) / BLOCK_M;
  const int cta_n_num = (N + BLOCK_N - 1) / BLOCK_N;

  /* - Split‑K 取值策略 -
   * - 经验：每个 block 处理不少于 ~32‑64 K 元素，否则 atomicAdd 性价⽐差
   * - 这里简单按 K 维 block 数 + 限⽌最⾼ 8 做启发式 */
  int split_k = max(1, (K + BLOCK_K - 1) / (BLOCK_K * 4));

  split_k = min(split_k, 32);  // safety cap
  // 确保 grid 维度 ≤ 2^31
  split_k = max(1, split_k);

  dim3 grid(split_k * cta_m_num * cta_n_num,  // X：Split‑K × M × N
            num_slices,                       // Y：slice
            max_active_loras);                // Z：LoRA 实例
  constexpr int TM = 2;
  constexpr int TN = 2;
  constexpr int THREAD_NUM = BLOCK_M * BLOCK_N / (TM * TN);
  dim3 block(THREAD_NUM);

  /* 连续性检查与 v1 保持一致 */
  if (!is_contiguous(lora_d0_stride, lora_d1_stride, lora_d2_stride) ||
      !is_contiguous(output_d0_stride, output_d1_stride, output_d2_stride) ||
      !is_contiguous(input_d0_stride, input_d1_stride)) {
    throw std::runtime_error(
        "lora_shrink_kernel_impl_v2: stride 参数必须是连续(contiguous)的");
  }
  /* 当 split_k > 1 时，为避免残余数据，与 host 侧约定先 memset(output, 0) */
  lora_shrink_kernel_v2<InputT, OutputT, TM, TN><<<grid, block, 0, stream>>>(
      input, lora_a_ptr_array, output, token_indices_sorted, lora_ids,
      num_tokens_per_lora, lora_token_start_loc, M, N, K, num_slices, scaling,
      input_d0_stride, input_d1_stride, lora_d0_stride, lora_d1_stride,
      lora_d2_stride, output_d0_stride, output_d1_stride, output_d2_stride);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error (Split‑K v2): %s\n",
           cudaGetErrorString(err));
  }
}

template <typename InputT, typename OutputT, int BLOCK_M = 64, int BLOCK_N = 64,
          int BLOCK_K = 32>
__global__ void lora_shrink_kernel_v3(
    const InputT* __restrict__ input,           // [num_tokens, hidden_size]
    const void* __restrict__ lora_a_ptr_array,  // slice -> weight ptr
    OutputT* __restrict__ output,  // [num_slices, num_tokens, lora_rank]
    const int* __restrict__ token_indices_sorted,
    const int* __restrict__ lora_ids,
    const int* __restrict__ num_tokens_per_lora,
    const int* __restrict__ lora_token_start_loc, int M, int N, int K,
    int num_slices, float scaling,
    // strides
    int input_d0_stride, int input_d1_stride, int lora_d0_stride,
    int lora_d1_stride, int lora_d2_stride, int output_d0_stride,
    int output_d1_stride, int output_d2_stride) {
  using namespace nvcuda;
  // 确定当前线程块负责的mn维度的总块数
  constexpr int padding = 8;
  const int cta_n_num = (N + BLOCK_N - 1) / BLOCK_N;
  const int cta_m_num = (M + BLOCK_M - 1) / BLOCK_M;
  const int SPLIT_K = gridDim.x / (cta_m_num * cta_n_num);  // 运行时可变

  const int pid_sk_m_n = blockIdx.x;
  // 按照split k分解
  const int pid_sk = pid_sk_m_n % SPLIT_K;
  // 除以split k后， 按照m维度分解
  const int cta_m_idx = (pid_sk_m_n / SPLIT_K) % cta_m_num;
  // 除以split k和m维度后，按照n维度分解
  const int cta_n_idx = pid_sk_m_n / (SPLIT_K * cta_m_num);

  const int slice_id = blockIdx.y;
  const int lora_idx = blockIdx.z;

  const int lora_id = lora_ids[lora_idx];
  if (lora_id < 0) return;  // 未启用 LoRA

  /* 获取当前 slice 的 A 权重指针 */
  const InputT* cur_lora_ptr = nullptr;
  if (num_slices == 1) {
    cur_lora_ptr = reinterpret_cast<const InputT*>(lora_a_ptr_array);
  } else {
    const int64_t* ptr_values =
        reinterpret_cast<const int64_t*>(lora_a_ptr_array);
    cur_lora_ptr = reinterpret_cast<const InputT*>(
        static_cast<uintptr_t>(ptr_values[slice_id]));
  }

  // 得到当前lora需要处理的token长度
  const int lora_m_size = num_tokens_per_lora[lora_idx];
  // 当前线程块负责的token的起始位置
  const int cta_m_offset = cta_m_idx * BLOCK_M;
  const int cta_n_offset = cta_n_idx * BLOCK_N;

  // 起始位置大于token长度，直接返回
  if (cta_m_offset >= lora_m_size || cta_n_offset >= N) return;
  const int lora_m_idx_start = lora_token_start_loc[lora_idx];

  // 当前线程块负责的K维度
  const int k_per_split = (K + SPLIT_K - 1) / SPLIT_K;
  const int k_begin = pid_sk * k_per_split;
  const int k_end = min(K, k_begin + k_per_split);

  // 共享内存 (为WMMA优化，使用swizzle访问，无padding)
  __shared__ InputT smem_input[BLOCK_M * (BLOCK_K + padding)];       // 行主序 (A)
  __shared__ InputT smem_lora_a[BLOCK_N * (BLOCK_K + padding)];      // 列主序 (B)
  __shared__ float smem_accumulator[BLOCK_M * BLOCK_N];  // 用于暂存WMMA结果

  // 线程与Warp坐标，用于WMMA
  const int warp_id = threadIdx.x >> 5;

  // WMMA fragment 尺寸 (16x16x16)
  constexpr int WM = 16, WN = 16, WK_FRAG = 16;
  // Warp 在 CTA 内的二维布局
  constexpr int WARPS_PER_ROW = BLOCK_N / WN;
  constexpr int TOTAL_WARPS = (BLOCK_M / WM) * WARPS_PER_ROW;

  // 当前 warp 在 CTA 内的二维坐标 (行, 列)
  const int warp_row_idx = warp_id / WARPS_PER_ROW;
  const int warp_col_idx = warp_id % WARPS_PER_ROW;

  // 初始化WMMA累加器 fragment
  wmma::fragment<wmma::accumulator, WM, WN, WK_FRAG, float> accumulator_frag;
  wmma::fill_fragment(accumulator_frag, 0.0f);

  // 主循环：遍历 K
  for (int k_off = k_begin; k_off < k_end; k_off += BLOCK_K) {
    // 载入input到共享内存 (行主序 + swizzle)
    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += blockDim.x) {
      const int lm = idx / BLOCK_K;
      const int lk = idx % BLOCK_K;
      InputT val = InputT(0);
      const int gm = cta_m_offset + lm;
      const int gk = k_off + lk;
      if (gm < lora_m_size && gk < k_end) {
        const int actual_token_idx =
            token_indices_sorted[lora_m_idx_start + gm];
        val = input[actual_token_idx * input_d0_stride + gk * input_d1_stride];
      }
      smem_input[lm * (BLOCK_K + padding) + lk] = val;
    }

    // 载入LoRA A到共享内存 (列主序 + swizzle)
    for (int idx = threadIdx.x; idx < BLOCK_N * BLOCK_K; idx += blockDim.x) {
      const int ln = idx / BLOCK_K;
      const int lk = idx % BLOCK_K;
      InputT val = InputT(0);
      const int gn = cta_n_offset + ln;
      const int gk = k_off + lk;
      if (gn < N && gk < k_end)
        val = cur_lora_ptr[lora_id * lora_d0_stride + gn * lora_d1_stride +
                           gk * lora_d2_stride];
      smem_lora_a[ln * (BLOCK_K + padding) + lk] = val;
    }
    __syncthreads();

    // 片上 WMMA 计算
    if (warp_id < TOTAL_WARPS) {
#pragma unroll
      for (int k_frag = 0; k_frag < BLOCK_K; k_frag += WK_FRAG) {
        const InputT* ptr_a = smem_input + warp_row_idx * WM * (BLOCK_K + padding) + k_frag;
        const InputT* ptr_b =
            smem_lora_a + warp_col_idx * WN * (BLOCK_K + padding) + k_frag;

        wmma::fragment<wmma::matrix_a, WM, WN, WK_FRAG, InputT, wmma::row_major>
            a_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK_FRAG, InputT, wmma::col_major>
            b_frag;

        wmma::load_matrix_sync(a_frag, ptr_a, BLOCK_K + padding);
        wmma::load_matrix_sync(b_frag, ptr_b, BLOCK_K + padding);
        wmma::mma_sync(accumulator_frag, a_frag, b_frag, accumulator_frag);
      }
    }
    __syncthreads();
  }

  // 将Warp的计算结果(fragment)写回到共享内存
  if (warp_id < TOTAL_WARPS) {
    wmma::store_matrix_sync(
        smem_accumulator + warp_row_idx * WM * BLOCK_N + warp_col_idx * WN,
        accumulator_frag, BLOCK_N, wmma::mem_row_major);
  }
  __syncthreads();

  // 写回（带 Split-K 归并）
  for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
    const int lm = idx / BLOCK_N;
    const int ln = idx % BLOCK_N;
    const int tok_in_lora = cta_m_offset + lm;
    const int rank_idx = cta_n_offset + ln;

    if (tok_in_lora < lora_m_size && rank_idx < N) {
      const int actual_tok =
          token_indices_sorted[lora_m_idx_start + tok_in_lora];
      const int out_off = slice_id * output_d0_stride +
                          actual_tok * output_d1_stride +
                          rank_idx * output_d2_stride;
      const float val = smem_accumulator[lm * BLOCK_N + ln] * scaling;

      // Split-K 归并
      if (SPLIT_K == 1) {
        output[out_off] = static_cast<OutputT>(val);
      } else {
        if constexpr (std::is_same_v<OutputT, float>) {
          atomicAdd(&output[out_off], val);
        } else if constexpr (std::is_same_v<OutputT, half>) {
          atomicAdd(&output[out_off], __float2half(val));
        } else if constexpr (std::is_same_v<OutputT, __nv_bfloat16>) {
          atomicAdd(&output[out_off], __float2bfloat16(val));
        }
      }
    }
  }
}

template <typename InT, typename OutT, int BM = 32, int BN = 32, int BK = 16>
void lora_shrink_kernel_impl_v3(
    const InT* input, const void* lora_ptrs, OutT* output,
    const int* tok_sorted, const int* lora_ids, const int* tok_cnt_per_lora,
    const int* lora_tok_start, int max_active_loras, int M, int N, int K,
    int num_slices, float scaling, int in_s0, int in_s1, int w_s0, int w_s1,
    int w_s2, int out_s0, int out_s1, int out_s2, cudaStream_t stream) {
  static_assert(std::is_same_v<InT, half> || std::is_same_v<InT, __nv_bfloat16>,
                "v3 WMMA only supports half/bfloat16 input");
  if (!is_contiguous(w_s0, w_s1, w_s2) ||
      !is_contiguous(out_s0, out_s1, out_s2) || !is_contiguous(in_s0, in_s1)) {
    throw std::runtime_error("strides must be contiguous");
  }

  int CTA_M = (M + BM - 1) / BM;
  int CTA_N = (N + BN - 1) / BN;
  int split_k = max(1, (K + BK - 1) / (BK * 4));
  split_k = min(split_k, 32);

  dim3 grid(split_k * CTA_M * CTA_N, num_slices, max_active_loras);
  dim3 block(512);
  // size_t shmem = (BM + BN) * BK * sizeof(InT) + BM * BN * sizeof(float);

  if (split_k > 1) {
    size_t bytes = static_cast<size_t>(num_slices) * M * N * sizeof(OutT);
    cudaMemsetAsync(output, 0, bytes, stream);
  }

  lora_shrink_kernel_v3<InT, OutT, BM, BN, BK><<<grid, block, 0, stream>>>(
      input, lora_ptrs, output, tok_sorted, lora_ids, tok_cnt_per_lora,
      lora_tok_start, M, N, K, num_slices, scaling, in_s0, in_s1, w_s0, w_s1,
      w_s2, out_s0, out_s1, out_s2);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("launch error (lora-v3-wmma): %s\n", cudaGetErrorString(err));
  }
}

void launch_lora_shrink_kernel(
    const void* input_ptr, const void* lora_a_ptr_array, void* output_ptr,
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
    lora_shrink_kernel_impl_v3<half, float>(
        static_cast<const half*>(input_ptr), lora_a_ptr_array,
        static_cast<float*>(output_ptr), token_indices_sorted_ptr, lora_ids_ptr,
        num_tokens_per_lora_ptr, lora_token_start_loc_ptr, max_active_loras, M,
        N, K, num_slices, scale, input_stride, 1,           // input strides
        lora_stride_0, lora_stride_1, lora_stride_2,        // lora strides
        output_stride_0, output_stride_1, output_stride_2,  // output strides
        stream);
  } else if (input_dtype == 0 && output_dtype == 0) {  // half -> half
    lora_shrink_kernel_impl_v3<half, half>(
        static_cast<const half*>(input_ptr), lora_a_ptr_array,
        static_cast<half*>(output_ptr), token_indices_sorted_ptr, lora_ids_ptr,
        num_tokens_per_lora_ptr, lora_token_start_loc_ptr, max_active_loras, M,
        N, K, num_slices, scale, input_stride, 1,           // input strides
        lora_stride_0, lora_stride_1, lora_stride_2,        // lora strides
        output_stride_0, output_stride_1, output_stride_2,  // output strides
        stream);
  } else if (input_dtype == 1 && output_dtype == 2) {  // bf16 -> float
    lora_shrink_kernel_impl_v3<__nv_bfloat16, float>(
        static_cast<const __nv_bfloat16*>(input_ptr), lora_a_ptr_array,
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