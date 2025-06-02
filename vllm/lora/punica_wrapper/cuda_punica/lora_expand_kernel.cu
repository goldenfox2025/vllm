/*
 * CUDA LoRA Expand Kernel
 *
 * Performs LoRA expand operation: output = input @ lora_b_weights
 * Supports multiple LoRA adapters and slice configurations
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
/*
 * LoRA Expand CUDA Kernel
 *
 * Parameters:
 * - input: Input tensor [num_slices, num_tokens, lora_rank]
 * - lora_b_ptr_array: Array of pointers to LoRA B weight tensors
 * - output: Output tensor [num_tokens, total_hidden_size]
 * - token_indices_sorted: Sorted token indices for each LoRA
 * - lora_ids: Active LoRA IDs
 * - num_tokens_per_lora: Number of tokens for each LoRA
 * - lora_token_start_loc: Starting location for each LoRA's tokens
 * - slice_starts: Starting position for each slice in output
 * - lora_strides_*: Memory strides for LoRA weight tensors
 * - hidden_sizes: Hidden size for each slice
 * - M: Number of tokens, MAX_N: Max hidden size, K: LoRA rank
 * - num_slices: Number of slices (e.g., Q/K/V)
 * - add_inputs: Whether to add to existing output values
 * - *_stride: Memory strides for input/output tensors
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
  // Calculate thread indices
  int pid_m = blockIdx.x % ((M + blockDim.y - 1) / blockDim.y);
  int pid_n = blockIdx.x / ((M + blockDim.y - 1) / blockDim.y);
  int slice_id = blockIdx.y;
  int lora_idx = blockIdx.z;

  int tid_m = threadIdx.y;
  int tid_n = threadIdx.x;

  // Boundary checks
  if (slice_id >= num_slices || lora_idx >= 3) return;

  int lora_id = lora_ids[lora_idx];
  if (lora_id == -1) return;
  // Calculate actual token and hidden indices
  int token_start = lora_token_start_loc[lora_idx];
  int num_tokens = num_tokens_per_lora[lora_idx];
  int token_offset = pid_m * blockDim.y + tid_m;

  if (token_offset >= num_tokens) return;

  int actual_token_idx = token_indices_sorted[token_start + token_offset];
  int hidden_offset = pid_n * blockDim.x + tid_n;
  int hidden_idx = hidden_offset;

  // Boundary checks using current slice's hidden size
  int current_slice_hidden_size = hidden_sizes[slice_id];
  if (hidden_idx >= current_slice_hidden_size) return;
  if (actual_token_idx >= M) return;

  // Get LoRA weight pointer for current slice
  const int64_t* ptr_values =
      reinterpret_cast<const int64_t*>(lora_b_ptr_array);
  uintptr_t ptr_value = static_cast<uintptr_t>(ptr_values[slice_id]);
  const InputT* cur_lora_ptr = reinterpret_cast<const InputT*>(ptr_value);

  // Get memory strides for current slice
  int cur_lora_d1_stride = lora_strides_d1[slice_id];
  int cur_lora_d2_stride = lora_strides_d2[slice_id];

  // Perform matrix multiplication: input[token, k] * weight[lora_id, hidden, k]
  float accumulator = 0.0f;

  // Main computation loop

  for (int k = 0; k < K; k++) {
    // Boundary checks
    if (lora_id < 0 || hidden_idx < 0 || k < 0 || actual_token_idx < 0)
      continue;

    // Calculate input offset (use slice_id for multi-slice scenarios)
    int input_offset = slice_id * input_d0_stride +
                       actual_token_idx * input_d1_stride + k * input_d2_stride;

    if (input_offset < 0) continue;
    InputT input_val = input[input_offset];

    // Calculate weight offset (cur_lora_ptr already points to specific slice)
    int weight_offset =
        hidden_idx * cur_lora_d1_stride + k * cur_lora_d2_stride;

    if (weight_offset < 0) continue;
    InputT lora_val = cur_lora_ptr[weight_offset];

    // Accumulate result
    float product =
        static_cast<float>(input_val) * static_cast<float>(lora_val);
    accumulator += product;
  }

  // Calculate output position
  int slice_start = slice_starts[slice_id];
  int output_hidden_idx = slice_start + hidden_idx;
  int output_offset = actual_token_idx * output_d0_stride +
                      output_hidden_idx * output_d1_stride;

  // Output boundary check
  if (output_offset < 0 || output_offset >= M * output_d0_stride) {
    return;
  }

  // Convert to output type
  OutputT result = static_cast<OutputT>(accumulator);

  // Add to existing values if required
  if (add_inputs) {
    OutputT existing_val = output[output_offset];
    float existing_float = static_cast<float>(existing_val);

    if (!isnan(existing_float) && !isinf(existing_float)) {
      result += existing_val;
    }
  }

  // Write output
  output[output_offset] = result;
}

// Template implementation function
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
  // Grid configuration
  const int BLOCK_M = 16;
  const int BLOCK_N = 16;

  int cta_m_num = (M + BLOCK_M - 1) / BLOCK_M;
  int cta_n_num = (MAX_N + BLOCK_N - 1) / BLOCK_N;

  dim3 grid(cta_m_num * cta_n_num, num_slices, max_active_loras);
  dim3 block(BLOCK_N, BLOCK_M);

  lora_expand_kernel<InputT, OutputT><<<grid, block, 0, stream>>>(
      input, lora_b_ptr_array, output, token_indices_sorted, lora_ids,
      num_tokens_per_lora, lora_token_start_loc, slice_starts, lora_strides_d0,
      lora_strides_d1, lora_strides_d2, hidden_sizes, M, MAX_N, K, num_slices,
      add_inputs, input_d0_stride, input_d1_stride, input_d2_stride,
      output_d0_stride, output_d1_stride);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA clean expand kernel launch error: %s\n",
           cudaGetErrorString(err));
  }
}

// Main launch function
void launch_lora_expand_kernel(
    const void* input_ptr, const void* lora_b_ptr, void* output_ptr,
    const int* token_indices_sorted_ptr, const int* lora_ids_ptr,
    const int* num_tokens_per_lora_ptr, const int* lora_token_start_loc_ptr,
    const int* slice_starts_ptr, const int* lora_strides_d0_ptr,
    const int* lora_strides_d1_ptr, const int* lora_strides_d2_ptr,
    const int* hidden_sizes_ptr, int max_active_loras,
    int num_total_tokens_in_batch, int lora_rank, int hidden_size,
    int num_slices, int offset_start, bool add_inputs, int input_stride0,
    int input_stride1, int input_stride2, int output_stride0,
    int output_stride1, cudaStream_t stream, int input_dtype,
    int output_dtype) {
  int M = num_total_tokens_in_batch;
  int MAX_N = hidden_size;
  int K = lora_rank;

  // Dispatch based on data types
  if (input_dtype == 2 && output_dtype == 1) {  // float -> bf16
    lora_expand_kernel_impl<float, __nv_bfloat16>(
        static_cast<const float*>(input_ptr), lora_b_ptr,
        static_cast<__nv_bfloat16*>(output_ptr), token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        slice_starts_ptr, lora_strides_d0_ptr, lora_strides_d1_ptr,
        lora_strides_d2_ptr, hidden_sizes_ptr, max_active_loras, M, MAX_N, K,
        num_slices, add_inputs, input_stride0, input_stride1, input_stride2,
        output_stride0, output_stride1, stream);
  } else if (input_dtype == 0 && output_dtype == 0) {  // fp16 -> fp16
    lora_expand_kernel_impl<__half, __half>(
        static_cast<const __half*>(input_ptr), lora_b_ptr,
        static_cast<__half*>(output_ptr), token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        slice_starts_ptr, lora_strides_d0_ptr, lora_strides_d1_ptr,
        lora_strides_d2_ptr, hidden_sizes_ptr, max_active_loras, M, MAX_N, K,
        num_slices, add_inputs, input_stride0, input_stride1, input_stride2,
        output_stride0, output_stride1, stream);
  } else if (input_dtype == 1 && output_dtype == 1) {  // bf16 -> bf16
    lora_expand_kernel_impl<__nv_bfloat16, __nv_bfloat16>(
        static_cast<const __nv_bfloat16*>(input_ptr), lora_b_ptr,
        static_cast<__nv_bfloat16*>(output_ptr), token_indices_sorted_ptr,
        lora_ids_ptr, num_tokens_per_lora_ptr, lora_token_start_loc_ptr,
        slice_starts_ptr, lora_strides_d0_ptr, lora_strides_d1_ptr,
        lora_strides_d2_ptr, hidden_sizes_ptr, max_active_loras, M, MAX_N, K,
        num_slices, add_inputs, input_stride0, input_stride1, input_stride2,
        output_stride0, output_stride1, stream);
  } else {
    std::cerr << "Clean kernel: Unsupported dtype combination: input="
              << input_dtype << ", output=" << output_dtype << std::endl;
  }
}
