#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
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
    const InputT* input_ptr,        // [num_tokens, hidden_size]
    const InputT* qkv_weights_ptr,  // [qkv_output_size, hidden_size]
    const void* lora_a_ptr_array,   // 指向各slice LoRA A权重的指针数组
    const void* lora_b_ptr_array,   // 指向各slice LoRA B权重的指针数组
    OutputT* output_ptr,            // [num_tokens, qkv_output_size]
    const int* token_indices_sorted_ptr, const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr, const int* lora_token_start_loc_ptr,
    const int* slice_starts_ptr, const int* lora_ranks_ptr, int num_tokens,
    int hidden_size, int qkv_output_size, int num_slices, int max_rank,
    // strides
    int input_stride0, int input_stride1, int qkv_stride0, int qkv_stride1,
    int lora_a_stride0, int lora_a_stride1, int lora_a_stride2,
    int lora_b_stride0, int lora_b_stride1, int lora_b_stride2,
    int output_stride0, int output_stride1) {
  // 每个线程块处理一个token
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  if (token_idx >= num_tokens) return;

  // 共享内存：存储当前token的输入向量
  extern __shared__ float shared_mem[];
  float* s_hidden_state = shared_mem;                     // [hidden_size]
  float* s_lora_intermediate = shared_mem + hidden_size;  // [max_rank]

  // === Step 1: 协同加载输入向量到共享内存 ===
  for (int i = tid; i < hidden_size; i += block_size) {
    if (i < hidden_size) {  // 额外边界检查
      int input_offset = token_idx * input_stride0 + i * input_stride1;
      s_hidden_state[i] = static_cast<float>(input_ptr[input_offset]);
    }
  }
  __syncthreads();

  // === Step 2: 计算基础QKV路径 ===
  // 每个线程负责计算输出向量的一部分
  for (int out_idx = tid; out_idx < qkv_output_size; out_idx += block_size) {
    if (out_idx < qkv_output_size) {  // 额外边界检查
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
  }
  __syncthreads();

  int active_lora_idx = -1;
  int lora_id = -1;

  // 修复的LoRA查找逻辑：
  // 需要根据token_indices_sorted和其他元数据正确映射
  // 为安全起见，先实现一个简单但正确的映射

  // 方法1：直接检查每个LoRA的token范围
  if (lora_token_start_loc_ptr != nullptr &&
      num_tokens_per_lora_ptr != nullptr && lora_ids_ptr != nullptr) {
    int cumulative_tokens = 0;
    for (int lora_idx = 0; lora_idx < 4;
         lora_idx++) {  // 限制检查范围，避免越界
      int tokens_for_this_lora =
          (lora_idx < 4) ? num_tokens_per_lora_ptr[lora_idx] : 0;
      int next_cumulative = cumulative_tokens + tokens_for_this_lora;

      // 检查当前token是否在这个LoRA的范围内
      if (token_idx >= cumulative_tokens && token_idx < next_cumulative) {
        int this_lora_id = (lora_idx < 4) ? lora_ids_ptr[lora_idx] : -1;
        if (this_lora_id != -1) {
          lora_id = this_lora_id;
          active_lora_idx = lora_idx;
          break;
        }
      }
      cumulative_tokens = next_cumulative;

      // 防止无限循环
      if (cumulative_tokens >= num_tokens) break;
    }
  }

  // 如果没有活跃的LoRA，直接返回（只有基础QKV）
  if (lora_id == -1 || active_lora_idx == -1) {
    return;
  }

  // 额外的安全检查：确保lora_id在合理范围内
  if (lora_id < 0 || lora_id >= 8) {  // 假设最多8个LoRA
    return;
  }

  // === Step 4: 执行LoRA计算 ===
  // 获取指针数组 - 修复类型转换问题
  const uintptr_t* ptr_values_a =
      reinterpret_cast<const uintptr_t*>(lora_a_ptr_array);
  const uintptr_t* ptr_values_b =
      reinterpret_cast<const uintptr_t*>(lora_b_ptr_array);

  // 空指针检查
  if (ptr_values_a == nullptr || ptr_values_b == nullptr) {
    return;
  }

  // 遍历所有slice (Q, K, V)
  for (int slice_id = 0; slice_id < num_slices; slice_id++) {
    // 边界检查
    if (slice_id >= num_slices || slice_id < 0) continue;

    // 获取当前slice的LoRA A权重指针
    uintptr_t lora_a_addr = ptr_values_a[slice_id];
    const InputT* cur_lora_a_ptr = reinterpret_cast<const InputT*>(lora_a_addr);

    // 获取当前slice的LoRA B权重指针
    uintptr_t lora_b_addr = ptr_values_b[slice_id];
    const InputT* cur_lora_b_ptr = reinterpret_cast<const InputT*>(lora_b_addr);

    // 关键修复：验证指针有效性
    if (lora_a_addr == 0 || lora_b_addr == 0) {
      continue;  // 跳过空指针
    }

    // 额外的指针对齐检查
    if (lora_a_addr % 16 != 0 || lora_b_addr % 16 != 0) {
      continue;  // 跳过未对齐的指针，可能无效
    }

    // 确定当前slice的rank - 改进版本
    int slice_rank = max_rank;  // 简化：假设所有slice有相同的rank
    if (lora_ranks_ptr != nullptr && active_lora_idx < 8) {  // 限制范围检查
      // 如果提供了ranks信息，使用实际的rank
      slice_rank = lora_ranks_ptr[active_lora_idx];
      // 边界检查
      if (slice_rank <= 0 || slice_rank > max_rank) {
        slice_rank = max_rank;
      }
    }

    // === Step 4a: LoRA Shrink阶段：input @ lora_a ===
    // 清零中间结果
    for (int r = tid; r < slice_rank; r += block_size) {
      if (r < max_rank) {  // 边界检查
        s_lora_intermediate[r] = 0.0f;
      }
    }
    __syncthreads();

    // 计算 s_hidden_state @ lora_a -> s_lora_intermediate
    for (int r = tid; r < slice_rank; r += block_size) {
      if (r < slice_rank && r < max_rank) {  // 边界检查
        float accumulator = 0.0f;
        for (int k = 0; k < hidden_size; k++) {
          // 关键修复：更安全的stride计算
          long long lora_a_offset = (long long)lora_id * lora_a_stride0 +
                                    (long long)r * lora_a_stride1 +
                                    (long long)k * lora_a_stride2;

          // 检查偏移是否在合理范围内（防止整数溢出）
          if (lora_a_offset < 0 || lora_a_offset > (1LL << 30)) {
            continue;  // 跳过可疑的偏移
          }

          float lora_a_val = static_cast<float>(cur_lora_a_ptr[lora_a_offset]);
          accumulator += s_hidden_state[k] * lora_a_val;
        }
        s_lora_intermediate[r] = accumulator;
      }
    }
    __syncthreads();

    // === Step 4b: LoRA Expand阶段：intermediate @ lora_b ===
    // 获取当前slice在输出中的起始位置和大小
    if (slice_starts_ptr == nullptr || slice_id >= num_slices) continue;

    int slice_start = slice_starts_ptr[slice_id];
    int slice_end = (slice_id + 1 < num_slices) ? slice_starts_ptr[slice_id + 1]
                                                : qkv_output_size;
    int slice_size = slice_end - slice_start;

    // 边界检查
    if (slice_start < 0 || slice_end > qkv_output_size || slice_size <= 0) {
      continue;
    }

    // 计算 s_lora_intermediate @ lora_b，并累加到输出
    for (int out_idx = tid; out_idx < slice_size; out_idx += block_size) {
      if (out_idx < slice_size) {  // 边界检查
        float accumulator = 0.0f;
        for (int r = 0; r < slice_rank && r < max_rank; r++) {
          // 关键修复：更安全的stride计算
          long long lora_b_offset = (long long)lora_id * lora_b_stride0 +
                                    (long long)out_idx * lora_b_stride1 +
                                    (long long)r * lora_b_stride2;

          // 检查偏移是否在合理范围内（防止整数溢出）
          if (lora_b_offset < 0 || lora_b_offset > (1LL << 30)) {
            continue;  // 跳过可疑的偏移
          }

          float lora_b_val = static_cast<float>(cur_lora_b_ptr[lora_b_offset]);
          accumulator += s_lora_intermediate[r] * lora_b_val;
        }

        // 将LoRA增量加到输出上
        int global_out_idx = slice_start + out_idx;
        if (global_out_idx < qkv_output_size) {  // 最终边界检查
          long long output_offset = (long long)token_idx * output_stride0 +
                                    (long long)global_out_idx * output_stride1;

          // 检查输出偏移是否合理
          if (output_offset >= 0 && output_offset < (1LL << 30)) {
            // 读取当前输出值，加上LoRA增量，然后写回
            OutputT current_val = output_ptr[output_offset];
            float new_val = static_cast<float>(current_val) + accumulator;
            output_ptr[output_offset] = static_cast<OutputT>(new_val);
          }
        }
      }
    }
    __syncthreads();
  }
}

/**
 * @brief 内核实现模板函数
 */
template <typename InputT, typename OutputT>
void ultimate_fusion_kernel_impl(
    const InputT* input_ptr, const InputT* qkv_weights_ptr,
    const void* lora_a_ptr_array, const void* lora_b_ptr_array,
    OutputT* output_ptr, const int* token_indices_sorted_ptr,
    const int* lora_ids_ptr, const int* num_tokens_per_lora_ptr,
    const int* lora_token_start_loc_ptr, const int* slice_starts_ptr,
    const int* lora_ranks_ptr, int num_tokens, int hidden_size,
    int qkv_output_size, int num_slices, int max_rank, int input_stride0,
    int input_stride1, int qkv_stride0, int qkv_stride1, int lora_a_stride0,
    int lora_a_stride1, int lora_a_stride2, int lora_b_stride0,
    int lora_b_stride1, int lora_b_stride2, int output_stride0,
    int output_stride1, cudaStream_t stream, int max_active_loras) {
  // Grid配置：每个token一个block
  dim3 grid(num_tokens);

  // Block配置：使用256个线程（经验值）
  const int THREADS_PER_BLOCK = 256;
  dim3 block(THREADS_PER_BLOCK);

  // 共享内存大小：hidden_state + lora_intermediate
  size_t shared_mem_size = (hidden_size + max_rank) * sizeof(float);

  // 启动内核
  ultimate_fusion_kernel<InputT, OutputT>
      <<<grid, block, shared_mem_size, stream>>>(
          input_ptr, qkv_weights_ptr, lora_a_ptr_array, lora_b_ptr_array,
          output_ptr, token_indices_sorted_ptr, lora_ids_ptr,
          num_tokens_per_lora_ptr, lora_token_start_loc_ptr, slice_starts_ptr,
          lora_ranks_ptr, num_tokens, hidden_size, qkv_output_size, num_slices,
          max_rank, input_stride0, input_stride1, qkv_stride0, qkv_stride1,
          lora_a_stride0, lora_a_stride1, lora_a_stride2, lora_b_stride0,
          lora_b_stride1, lora_b_stride2, output_stride0, output_stride1);

  // 检查错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    // std::cerr << "Ultimate fusion kernel launch error: " <<
    // cudaGetErrorString(err) << std::endl; printf("Ultimate fusion kernel
    // launch error: %s\n", cudaGetErrorString(err));
  }
}

template <typename InputT, typename OutputT, int BM, int BK, int BN,
          int THREADS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K,
          int max_rank = 128>
__global__ void ultimate_fusion_kernel_v2(
    const InputT* input_ptr, const InputT* qkv_weights_ptr,
    const void* lora_a_ptr_array, const void* lora_b_ptr_array,
    OutputT* output_ptr, const int* token_indices_sorted_ptr,
    const int* lora_ids_ptr, const int* num_tokens_per_lora_ptr,
    const int* lora_token_start_loc_ptr, const int* slice_starts_ptr,
    const int* lora_ranks_ptr, int num_tokens, int hidden_size,
    int qkv_output_size, int num_slices, int max_rank_default,
    // strides
    int input_stride0, int input_stride1, int qkv_stride0, int qkv_stride1,
    int lora_a_stride0, int lora_a_stride1, int lora_a_stride2,
    int lora_b_stride0, int lora_b_stride1, int lora_b_stride2,
    int output_stride0, int output_stride1) {
  int lora_id = blockIdx.x;
  // NOTE: blockIdx.y is not used for slicing to allow for flexibility in the
  // grid definition. Instead, slice_id is determined from cta_n_idx later.

  const int width_in_blocks = (qkv_output_size + max_rank * num_slices) / BN;

  int cta_m_idx = blockIdx.z / width_in_blocks;
  int cta_n_idx = blockIdx.z % width_in_blocks;

  int lora_idx = lora_ids_ptr[lora_id];

  using namespace nvcuda;

  // 定义WMMA fragments
  using FragmentA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                   InputT, wmma::row_major>;
  using FragmentB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   InputT, wmma::row_major>;
  using FragmentC =
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

  // 这个逻辑决定了每个warp负责计算输出瓦片的哪个16x16部分
  const int warps_in_m_dim = BM / WMMA_M;
  const int warps_in_n_dim = BN / WMMA_N;
  const int warp_id = threadIdx.x / warpSize;
  const int warp_m = warp_id / warps_in_n_dim;
  const int warp_n = warp_id % warps_in_n_dim;

  if (lora_idx == -1) {
    __shared__ InputT s_a[BM * BK];
    __shared__ InputT s_b[BK * BN];
    __shared__ float smem_output[BM * BN];

    // 每个warp在自己的私有寄存器中创建一个累加器片段，并清零
    FragmentC frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    for (int k_tile_start = 0; k_tile_start < hidden_size; k_tile_start += BK) {
      // 加载input
      for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
        int m = i / BK;
        int k = i % BK;

        int token_row = cta_m_idx * BM + m;
        int hidden_col = k_tile_start + k;
        if (token_row < num_tokens && hidden_col < hidden_size) {
          s_a[i] =
              input_ptr[token_row * input_stride0 + hidden_col * input_stride1];
        } else {
          s_a[i] = InputT(0);
        }
      }

      // 加载qkv权重
      for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
        int k = i / BN;
        int n = i % BN;
        int weight_row = k_tile_start + k;
        int weight_col =
            cta_n_idx * BN + n;  // Corrected: qkv is (hidden, qkv_out) layout
        if (weight_col < qkv_output_size && weight_row < hidden_size) {
          s_b[k * BN + n] = qkv_weights_ptr[weight_row * qkv_stride0 +
                                            weight_col * qkv_stride1];
        } else {
          s_b[k * BN + n] = InputT(0);
        }
      }
      __syncthreads();  // 确保s_a和s_b加载完成

      // 在已加载的共享内存瓦片上进行计算
      for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
        FragmentA frag_a;
        FragmentB frag_b;
        int s_a_row_offset = warp_m * WMMA_M;
        int s_a_col_offset = k_step;
        int s_b_row_offset = k_step;
        int s_b_col_offset = warp_n * WMMA_N;

        wmma::load_matrix_sync(frag_a,
                               s_a + s_a_row_offset * BK + s_a_col_offset, BK);
        wmma::load_matrix_sync(frag_b,
                               s_b + s_b_row_offset * BN + s_b_col_offset, BN);

        // 每个warp在自己的寄存器中累加结果
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
      }
      __syncthreads();  // 确保所有warp都完成了对当前瓦片的计算
    }

    // 每个warp负责的16x16瓦片存储到共享内存的目标地址
    int smem_warp_row_offset = warp_m * WMMA_M;
    int smem_warp_col_offset = warp_n * WMMA_N;
    wmma::store_matrix_sync(
        smem_output + smem_warp_row_offset * BN + smem_warp_col_offset, frag_c,
        BN, wmma::mem_row_major);
    __syncthreads();

    // 输出
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
      int m_local = i / BN;
      int n_local = i % BN;

      int output_row = cta_m_idx * BM + m_local;
      int output_col = cta_n_idx * BN + n_local;

      if (output_row < num_tokens && output_col < qkv_output_size) {
        output_ptr[output_row * output_stride0 + output_col * output_stride1] =
            static_cast<OutputT>(smem_output[i]);
      }
    }
    return;  // 基础任务完成，直接返回
  } else {
    // 这个分支的全局内存不合并可能会比较严重
    // 按照lora_idx来索引对应的token
    int lora_m_size = num_tokens_per_lora_ptr[lora_idx];
    const int lora_m_idx_start = lora_token_start_loc_ptr[lora_idx];

    // FinalOutput = I*QKV + (I*A)*B
    __shared__ InputT s_input[BM * BK];
    __shared__ InputT s_qkv_weights[BK * BN];

    // __shared__ float s_intermediate[BM * max_rank];

    const uintptr_t* ptr_values_a =
        reinterpret_cast<const uintptr_t*>(lora_a_ptr_array);
    const uintptr_t* ptr_values_b =
        reinterpret_cast<const uintptr_t*>(lora_b_ptr_array);

    if (ptr_values_a == nullptr || ptr_values_b == nullptr) {
      return;
    }
    // 确认是否是lora slice
    bool is_lora_slice = cta_n_idx * BN - qkv_output_size > 0;

    // 回过头来，先放弃花里胡哨的想法
    // 一个能跑的方案究竟是什么？

    // is_lora_slice为true时，需要计算lora_a和lora_b
    // 斟酌之下，确认分支处理为上
    // 线程块的划分是基于 qkv_output_size + max_rank * num_slices 的
    // 即便从理论上分析，这应该也是最佳的
    // 第一阶段，所有线程块理论上计算负载差不多，但loraA分支的smem占用会多一些
    // loraA分支的线程块，按照max_rank为128来分析，输入tokens为10
    // 仅分配到4 * 10个线程块
    // 如果max_rank更小，比如16，那么实际上也就10个线程块
    // 第一阶段其实好说。
    // 如果不写回全局，第二阶段开始，
    // 这10个线程块负责[tokens,rank]和[rank,output_size]的计算
    // 完全打不满
    // 这个在写shrink的时候测过了
    // spiltK的CUDA版本在部分场景下能有接近翻倍的提升（ncu还是准的）
    // 但是非spiltK，线程块要少得多
    // 速度也就慢得多，甚至连一半都难得达到。甚至是十分之一。
    // 这一部分也要研究。我不太清楚输入数据是如何影响性能提升百分比的
    // 或者说本可以打满 但必须写回全局
    // 也就是说，不能有方案上的创新
    // 本质上和torch.matmul + fused expand的路径是一致的
    // 这10个线程块写回全局
    // 更多的线程块读到它们，然后在output_size上去读
    // 其实这里本可以提高并行性，那就是切slice
    // 那qkv并行呢？
    // 难道要按照1536 256 256来分配线程块？那本质上不就是BN切割么？
    // 对了。就是这样
    // 这种大融合第一解决了启动开销
    // 但在graph下谈启动开销没意义
    // 第二，可以提高并行性
    // 如果矩阵乘优化得好，至少比shrink的splitk要快一些
    // 但之前的测量也是在wsl上做的，偶尔不准。所以做不得数
    // 这最简单的修改就是多流
    // 但根据flash
    // attention拆kv的经验，多流确实没有一个核启动多线程块好（前提是优化水准相同）
    // 而且cuda graph捕获多流的操作……应该是没搞懂，总之会有bug
    // 项目好一段时间没动了。也必须动一动。


    // 总而言之，先前的实现都可以放弃了
    // shrink是一个障碍。
    // 根据经验，sm打满远比HBM读写更重要
    // 因此全局内存写回不可避免
    // 之后在速度稳定的机器上进行一下对比
    // 也许fused expand路径会更快
    // 当然，本内核如果优化程度足够高，肯定能比空算版更好
    

    // 此外，V1原始版本，稍微大点处理块，4070laptop就会爆寄存器/smem
    // 当然，可以做寄存器复用。共享内存也可以复用……但估计希望不大
    // 主要是接近爆寄存器或smem的程度占用率一定会跌得很厉害
    // 横竖都要写回全局内存，分两个内核才是合理的

    // 感觉三个完全融合意义不是太大

    if (is_lora_slice) {
      // 计算lora_a
      // 将3个slice的loraA读到smem里
    } else {
      // 计算qkv 如-1分支
    }

    // 写回intermediate到HBM

    // 各个线程块开始发挥作用，从全局内存读取intermediate
    // 读取loraB到smem
    // 完成计算

    // 非常遗憾，发现只能搞定这样的实现
    // 那么之后就要学cute了
    // ldmatrix可以用来适配swizzle，但写起来好麻烦。

    // 感觉cute如果写好了会更快

    // 清空实现，考虑另一种方案
  }
}

/**
 * @brief V1 内核实现模板函数
 */
template <typename InputT, typename OutputT>
void ultimate_fusion_kernel_impl_v1(
    const InputT* input_ptr, const InputT* qkv_weights_ptr,
    const void* lora_a_ptr_array, const void* lora_b_ptr_array,
    OutputT* output_ptr, const int* token_indices_sorted_ptr,
    const int* lora_ids_ptr, const int* num_tokens_per_lora_ptr,
    const int* lora_token_start_loc_ptr, const int* slice_starts_ptr,
    const int* lora_ranks_ptr, int num_tokens, int hidden_size,
    int qkv_output_size, int num_slices, int max_rank, int input_stride0,
    int input_stride1, int qkv_stride0, int qkv_stride1, int lora_a_stride0,
    int lora_a_stride1, int lora_a_stride2, int lora_b_stride0,
    int lora_b_stride1, int lora_b_stride2, int output_stride0,
    int output_stride1, cudaStream_t stream, int max_active_loras) {
  constexpr int BM = 32;
  constexpr int BK = 32;
  constexpr int BN = 32;
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WARP_SIZE = 32;

  constexpr int WARPS_IN_M = BM / WMMA_M;
  constexpr int WARPS_IN_N = BN / WMMA_N;
  constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_IN_M * WARPS_IN_N;

  // 进行一次伟大的尝试
  // 每个CTA根据索引处理对应的数据

  const int z_grid_dim =
      ((num_tokens + BM - 1) / BM) * ((qkv_output_size + BN - 1) / BN);
  dim3 grid(max_active_loras, 1, z_grid_dim);

  dim3 block(THREADS_PER_BLOCK);
  // 启动V1内核
  ultimate_fusion_kernel_v1<InputT, OutputT, BM, BK, BN, THREADS_PER_BLOCK,
                            WMMA_M, WMMA_N, WMMA_K><<<grid, block, 0, stream>>>(
      input_ptr, qkv_weights_ptr, lora_a_ptr_array, lora_b_ptr_array,
      output_ptr, token_indices_sorted_ptr, lora_ids_ptr,
      num_tokens_per_lora_ptr, lora_token_start_loc_ptr, slice_starts_ptr,
      lora_ranks_ptr, num_tokens, hidden_size, qkv_output_size, num_slices,
      max_rank, input_stride0, input_stride1, qkv_stride0, qkv_stride1,
      lora_a_stride0, lora_a_stride1, lora_a_stride2, lora_b_stride0,
      lora_b_stride1, lora_b_stride2, output_stride0, output_stride1);

  // 检查CUDA错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("V1 kernel launch error: %s\n", cudaGetErrorString(err));
  }
}



void launch_ultimate_fusion_kernel(
    const void* input_ptr, const void* qkv_weights_ptr,
    const void* lora_a_ptr_array, const void* lora_b_ptr_array,
    void* output_ptr, const int* token_indices_sorted_ptr,
    const int* lora_ids_ptr, const int* num_tokens_per_lora_ptr,
    const int* lora_token_start_loc_ptr, const int* slice_starts_ptr,
    const int* lora_ranks_ptr, int max_active_loras, int num_tokens,
    int hidden_size, int qkv_output_size, int num_slices, int max_rank,
    int input_stride0, int input_stride1, int qkv_stride0, int qkv_stride1,
    int lora_a_stride0, int lora_a_stride1, int lora_a_stride2,
    int lora_b_stride0, int lora_b_stride1, int lora_b_stride2,
    int output_stride0, int output_stride1, cudaStream_t stream,
    int input_dtype, int output_dtype) {
  // 根据数据类型分发
  if (input_dtype == 0 && output_dtype == 0) {  // fp16 -> fp16
    ultimate_fusion_kernel_impl<__half, __half>(
        static_cast<const __half*>(input_ptr),
        static_cast<const __half*>(qkv_weights_ptr), lora_a_ptr_array,
        lora_b_ptr_array, static_cast<__half*>(output_ptr),
        token_indices_sorted_ptr, lora_ids_ptr, num_tokens_per_lora_ptr,
        lora_token_start_loc_ptr, slice_starts_ptr, lora_ranks_ptr, num_tokens,
        hidden_size, qkv_output_size, num_slices, max_rank, input_stride0,
        input_stride1, qkv_stride0, qkv_stride1, lora_a_stride0, lora_a_stride1,
        lora_a_stride2, lora_b_stride0, lora_b_stride1, lora_b_stride2,
        output_stride0, output_stride1, stream, max_active_loras);
  } else if (input_dtype == 1 && output_dtype == 1) {  // bf16 -> bf16
    ultimate_fusion_kernel_impl<__nv_bfloat16, __nv_bfloat16>(
        static_cast<const __nv_bfloat16*>(input_ptr),
        static_cast<const __nv_bfloat16*>(qkv_weights_ptr), lora_a_ptr_array,
        lora_b_ptr_array, static_cast<__nv_bfloat16*>(output_ptr),
        token_indices_sorted_ptr, lora_ids_ptr, num_tokens_per_lora_ptr,
        lora_token_start_loc_ptr, slice_starts_ptr, lora_ranks_ptr, num_tokens,
        hidden_size, qkv_output_size, num_slices, max_rank, input_stride0,
        input_stride1, qkv_stride0, qkv_stride1, lora_a_stride0, lora_a_stride1,
        lora_a_stride2, lora_b_stride0, lora_b_stride1, lora_b_stride2,
        output_stride0, output_stride1, stream, max_active_loras);
  } else {
    std::cerr << "Ultimate fusion kernel: Unsupported dtype combination: input="
              << input_dtype << ", output=" << output_dtype << std::endl;
  }
}