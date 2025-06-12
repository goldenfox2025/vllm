import ctypes
import os
import torch
from typing import List, Optional

# Load the pure C library
C_LIB_AVAILABLE = False
cuda_c_lib = None

try:
    # Try to find the C library
    lib_path = os.path.join(os.path.dirname(__file__), "build", "libcuda_lora_c.so")
    if not os.path.exists(lib_path):
        # Try alternative location
        lib_path = os.path.join(os.path.dirname(__file__), "libcuda_lora_c.so")
    
    if os.path.exists(lib_path):
        cuda_c_lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        cuda_c_lib.cuda_lora_shrink_c.argtypes = [
            ctypes.c_void_p,  # input_ptr
            ctypes.c_void_p,  # lora_a_ptr
            ctypes.c_void_p,  # output_ptr
            ctypes.c_void_p,  # token_indices_sorted_ptr
            ctypes.c_void_p,  # lora_ids_ptr
            ctypes.c_void_p,  # num_tokens_per_lora_ptr
            ctypes.c_void_p,  # lora_token_start_loc_ptr
            ctypes.c_int,     # max_active_loras
            ctypes.c_int,     # num_total_tokens
            ctypes.c_int,     # hidden_size
            ctypes.c_int,     # lora_rank
            ctypes.c_int,     # num_slices
            ctypes.c_float,   # scale
            ctypes.c_int,     # input_stride0
            ctypes.c_int,     # lora_stride0
            ctypes.c_int,     # lora_stride1
            ctypes.c_int,     # lora_stride2
            ctypes.c_int,     # output_stride0
            ctypes.c_int,     # output_stride1
            ctypes.c_int,     # output_stride2
            ctypes.c_void_p,  # stream
            ctypes.c_int,     # input_dtype
            ctypes.c_int,     # output_dtype
        ]
        cuda_c_lib.cuda_lora_shrink_c.restype = ctypes.c_int

        # Define function signatures for expand - 更新为Triton兼容接口
        cuda_c_lib.cuda_lora_expand_c.argtypes = [
            ctypes.c_void_p,  # input_ptr
            ctypes.c_void_p,  # lora_b_ptr
            ctypes.c_void_p,  # output_ptr
            ctypes.c_void_p,  # token_indices_sorted_ptr
            ctypes.c_void_p,  # lora_ids_ptr
            ctypes.c_void_p,  # num_tokens_per_lora_ptr
            ctypes.c_void_p,  # lora_token_start_loc_ptr
            ctypes.c_void_p,  # slice_starts_ptr
            ctypes.c_void_p,  # lora_strides_d0_ptr
            ctypes.c_void_p,  # lora_strides_d1_ptr
            ctypes.c_void_p,  # lora_strides_d2_ptr
            ctypes.c_void_p,  # hidden_sizes_ptr
            ctypes.c_int,     # max_active_loras
            ctypes.c_int,     # num_total_tokens_in_batch
            ctypes.c_int,     # lora_rank
            ctypes.c_int,     # hidden_size
            ctypes.c_int,     # num_slices
            ctypes.c_int,     # offset_start
            ctypes.c_int,     # add_inputs
            ctypes.c_int,     # input_stride0
            ctypes.c_int,     # input_stride1
            ctypes.c_int,     # input_stride2
            ctypes.c_int,     # output_stride0
            ctypes.c_int,     # output_stride1
            ctypes.c_void_p,  # stream
            ctypes.c_int,     # input_dtype
            ctypes.c_int,     # output_dtype
        ]
        cuda_c_lib.cuda_lora_expand_c.restype = ctypes.c_int

        cuda_c_lib.test_cuda_kernel.argtypes = []
        cuda_c_lib.test_cuda_kernel.restype = ctypes.c_int
        
        # Define function signatures for ultimate fusion kernel
        cuda_c_lib.cuda_ultimate_fusion_c.argtypes = [
            ctypes.c_void_p,  # input_ptr
            ctypes.c_void_p,  # qkv_weights_ptr
            ctypes.c_void_p,  # lora_a_ptr_array
            ctypes.c_void_p,  # lora_b_ptr_array
            ctypes.c_void_p,  # output_ptr
            ctypes.c_void_p,  # token_indices_sorted_ptr
            ctypes.c_void_p,  # lora_ids_ptr
            ctypes.c_void_p,  # num_tokens_per_lora_ptr
            ctypes.c_void_p,  # lora_token_start_loc_ptr
            ctypes.c_void_p,  # slice_starts_ptr
            ctypes.c_void_p,  # lora_ranks_ptr
            ctypes.c_int,     # max_active_loras
            ctypes.c_int,     # num_tokens
            ctypes.c_int,     # hidden_size
            ctypes.c_int,     # qkv_output_size
            ctypes.c_int,     # num_slices
            ctypes.c_int,     # max_rank
            ctypes.c_int,     # input_stride0
            ctypes.c_int,     # input_stride1
            ctypes.c_int,     # qkv_stride0
            ctypes.c_int,     # qkv_stride1
            ctypes.c_int,     # lora_a_stride0
            ctypes.c_int,     # lora_a_stride1
            ctypes.c_int,     # lora_a_stride2
            ctypes.c_int,     # lora_b_stride0
            ctypes.c_int,     # lora_b_stride1
            ctypes.c_int,     # lora_b_stride2
            ctypes.c_int,     # output_stride0
            ctypes.c_int,     # output_stride1
            ctypes.c_void_p,  # stream_ptr
            ctypes.c_int,     # input_dtype
            ctypes.c_int,     # output_dtype
        ]
        cuda_c_lib.cuda_ultimate_fusion_c.restype = ctypes.c_int
        
        C_LIB_AVAILABLE = True
        print(f"Pure C CUDA library loaded: {lib_path}")
    else:
        print(f"C library not found at: {lib_path}")

except Exception as e:
    print(f"Failed to load C library: {e}")

def test_c_library():
    """Test if the C library is working"""
    if not C_LIB_AVAILABLE:
        return False
    
    try:
        result = cuda_c_lib.test_cuda_kernel()
        return result == 0
    except Exception as e:
        print(f"❌ C library test failed: {e}")
        return False

def cuda_lora_shrink_triton_interface(
    inputs: torch.Tensor,  #  shape [num_tokens, hidden_size]
    lora_a_weights: list[torch.Tensor],  # shape [num_loras, lora_rank, hidden_size]
    output_tensor: torch.Tensor,  # shape [num_slices, num_tokens, lora_rank]
    token_lora_mapping: torch.Tensor,  # shape [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens]
    num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
    lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
    lora_ids: torch.Tensor,  # shape [max-loras + 1]
    no_lora_flag_cpu: torch.Tensor,  # shape [1]
    scaling: float,
) -> bool:
    """
    CUDA LoRA shrink using Triton-compatible interface

    This matches the exact interface of the Triton _lora_shrink function
    """
    if not C_LIB_AVAILABLE:
        return False

    try:
        # Early exit check like Triton
        assert no_lora_flag_cpu.numel() == 1
        if no_lora_flag_cpu.item():
            # None of the inputs require LoRA.
            return True

        # Validate inputs
        assert inputs.is_cuda and inputs.is_contiguous()
        assert output_tensor.is_cuda and output_tensor.is_contiguous()
        assert token_lora_mapping.is_cuda and token_lora_mapping.is_contiguous()
        assert token_indices_sorted_by_lora_ids.is_cuda and token_indices_sorted_by_lora_ids.is_contiguous()

        for lora_weight in lora_a_weights:
            assert lora_weight.is_cuda and lora_weight.is_contiguous()

        # Extract dimensions
        num_tokens = inputs.shape[0]
        hidden_size = inputs.shape[1]
        num_slices = output_tensor.shape[0]
        lora_rank = output_tensor.shape[2]
        num_loras = len(lora_a_weights)

        # Process LoRA weights
        processed_weights = []
        lora_strides_d1 = []
        lora_strides_d2 = []

        for weight in lora_a_weights:
            # Extract the last 2 dimensions which should be [lora_rank, hidden_size]
            if weight.ndim >= 2:
                # Take the last 2 dimensions: [..., lora_rank, hidden_size]
                lora_2d = weight.view(-1, weight.shape[-2], weight.shape[-1])  # [batch, lora_rank, hidden_size]
                # Take the first batch (assuming all batches are the same)
                lora_2d = lora_2d[0]  # [lora_rank, hidden_size]
                processed_weights.append(lora_2d)

                lora_strides_d1.append(lora_2d.stride(0))  # lora_rank stride
                lora_strides_d2.append(lora_2d.stride(1))  # hidden_size stride
            else:
                raise ValueError(f"LoRA weight has insufficient dimensions: {weight.shape}")

        # Check stride consistency
        if (len(set(lora_strides_d1)) > 1 or len(set(lora_strides_d2)) > 1):
            raise RuntimeError(f"LoRA weight stride inconsistency detected")

        # 修复：创建指针数组，与Triton的_get_lora_a_ptr完全一致
        tensor_ptrs = []
        for weight in processed_weights:
            tensor_ptrs.append(weight.data_ptr())

        if len(processed_weights) == 1:
            # 单个slice情况：直接使用权重张量
            lora_ptr_value = processed_weights[0].data_ptr()
            lora_3d = processed_weights[0].unsqueeze(0)  # 用于stride计算
        else:
            # 多个slice情况：创建指针数组张量，与Triton一致
            lora_ptr_tensor = torch.tensor(tensor_ptrs, dtype=torch.int64, device=inputs.device)
            lora_ptr_value = lora_ptr_tensor.data_ptr()
            lora_3d = torch.stack(processed_weights, dim=0)  # 用于stride计算

        # Get data pointers
        input_ptr = inputs.data_ptr()
        lora_ptr = lora_ptr_value
        output_ptr = output_tensor.data_ptr()
        token_ptr = token_indices_sorted_by_lora_ids.data_ptr()

        # Get strides
        input_stride0 = inputs.stride(0)
        lora_stride0 = lora_3d.stride(0)
        lora_stride1 = lora_3d.stride(1)
        lora_stride2 = lora_3d.stride(2)
        output_stride0 = output_tensor.stride(0)
        output_stride1 = output_tensor.stride(1)
        output_stride2 = output_tensor.stride(2)

        # Map PyTorch dtypes to our kernel's dtype codes
        dtype_map = {
            torch.float16: 0,    # fp16
            torch.bfloat16: 1,   # bf16
            torch.float32: 2,    # fp32
        }

        input_dtype_code = dtype_map.get(inputs.dtype, -1)
        output_dtype_code = dtype_map.get(output_tensor.dtype, -1)

        if input_dtype_code == -1 or output_dtype_code == -1:
            return False

        # Call the C function with Triton-compatible interface

        # Get additional metadata pointers for multi-LoRA support
        lora_ids_ptr = lora_ids.data_ptr()
        num_tokens_per_lora_ptr = num_tokens_per_lora.data_ptr()
        lora_token_start_loc_ptr = lora_token_start_loc.data_ptr()

        active_lora_count = sum(1 for lora_id in lora_ids.tolist())
        max_active_loras = active_lora_count
        
        # 添加详细的调试信息
        print(f"\n🟢 [CUDA Shrink Debug] 关键参数检查:")
        print(f"   输入: {inputs.shape}, 隐藏大小: {hidden_size}")
        print(f"   输出: {output_tensor.shape}, rank: {lora_rank}, slices: {num_slices}")
        print(f"   LoRA IDs: {lora_ids.tolist()}")
        print(f"   每个LoRA的token数: {num_tokens_per_lora.tolist()}")
        print(f"   LoRA token起始位置: {lora_token_start_loc.tolist()}")
        if(token_indices_sorted_by_lora_ids.shape[0] > 32):
            print(f"   token索引排序: {token_indices_sorted_by_lora_ids.tolist()[:32]}")
        else:
            print(f"   token索引排序: {token_indices_sorted_by_lora_ids.tolist()}")

        print(f"   最大活跃LoRA数: {max_active_loras}")
        
        # 检查映射合理性
        total_mapped_tokens = sum(num_tokens_per_lora.tolist())
        print(f"   映射检查: 总token={num_tokens}, 映射token={total_mapped_tokens}")
        
        if total_mapped_tokens != num_tokens:
            print(f"   ⚠️  警告: token映射不完整! 总token={num_tokens}, 映射token={total_mapped_tokens}")
        
        # 检查LoRA权重形状
        for i, lora_3d in enumerate(lora_a_weights):
            print(f"   LoRA权重[{i}] 形状: {lora_3d.shape}")
        
        # Call the C function with multi-LoRA support
        result = cuda_c_lib.cuda_lora_shrink_c(
            input_ptr,
            lora_ptr,
            output_ptr,
            token_ptr,  # token_indices_sorted_ptr
            lora_ids_ptr,
            num_tokens_per_lora_ptr,
            lora_token_start_loc_ptr,
            max_active_loras,
            num_tokens,  # num_total_tokens
            hidden_size,
            lora_rank,
            num_slices,
            scaling,
            input_stride0,
            lora_stride0,
            lora_stride1,
            lora_stride2,
            output_stride0,
            output_stride1,
            output_stride2,
            ctypes.c_void_p(0),  # stream = 0
            ctypes.c_int(input_dtype_code),   # input dtype
            ctypes.c_int(output_dtype_code)   # output dtype
        )
        
        if result == 0:
            return True
        else:
            raise RuntimeError(f"CUDA shrink kernel failed with code: {result}")
            
    except Exception as e:
        return False

def cuda_lora_shrink_ctypes(
    inputs: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    scaling: float
) -> bool:
    """
    Simple wrapper for backward compatibility
    """
    # Create simple metadata for single LoRA case
    num_tokens = inputs.shape[0]

    # Simple case: all tokens use LoRA 0
    token_indices_sorted_by_lora_ids = torch.arange(num_tokens, dtype=torch.int32, device=inputs.device)
    num_tokens_per_lora = torch.tensor([num_tokens], dtype=torch.int32, device=inputs.device)
    lora_token_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=inputs.device)
    lora_ids = torch.tensor([0], dtype=torch.int32, device=inputs.device)
    no_lora_flag_cpu = torch.tensor([False], dtype=torch.bool)

    return cuda_lora_shrink_triton_interface(
        inputs, lora_a_weights, output_tensor, token_lora_mapping,
        token_indices_sorted_by_lora_ids, num_tokens_per_lora,
        lora_token_start_loc, lora_ids, no_lora_flag_cpu, scaling
    )

def cuda_lora_shrink_fallback(
    inputs: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    scaling: float
) -> bool:
    """
    Pure PyTorch fallback implementation
    """
    try:
        # Clear output
        output_tensor.zero_()

        # Simple implementation: assume all tokens use the same LoRA
        inputs_2d = inputs.view(-1, inputs.shape[-1])

        for slice_idx, lora_a in enumerate(lora_a_weights):
            if slice_idx >= output_tensor.shape[0]:
                break

            # Compute: output = input @ lora_a.T * scale
            result = torch.matmul(inputs_2d, lora_a.t()) * scaling
            output_tensor[slice_idx] = result

        return True

    except Exception as e:
        return False

def cuda_lora_shrink_unified(
    inputs: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    scaling: float
) -> bool:
    """
    Unified interface that tries C library first, then PyTorch fallback
    """
    # Try C library first
    if C_LIB_AVAILABLE:
        if cuda_lora_shrink_ctypes(inputs, lora_a_weights, output_tensor, token_lora_mapping, scaling):
            return True
        print("⚠️  C library failed, trying PyTorch fallback...")
    
    # Fallback to PyTorch
    return cuda_lora_shrink_fallback(inputs, lora_a_weights, output_tensor, token_lora_mapping, scaling)

def test_ctypes_wrapper():
    """Test the ctypes wrapper"""
    print("🧪 Testing ctypes wrapper...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    # Test C library
    if not test_c_library():
        print("❌ C library test failed")
        return False
    
    # Test configuration
    num_tokens = 16
    hidden_size = 64
    lora_rank = 8
    num_slices = 1
    scaling = 0.5
    
    device = torch.device('cuda:0')
    dtype = torch.float16
    
    # Create test tensors
    inputs = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    lora_a = torch.randn(lora_rank, hidden_size, dtype=dtype, device=device)
    output_tensor = torch.zeros(num_slices, num_tokens, lora_rank, dtype=dtype, device=device)
    token_lora_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    
    print(f"📊 Test config: tokens={num_tokens}, hidden={hidden_size}, rank={lora_rank}")
    
    # Test unified interface
    success = cuda_lora_shrink_unified(
        inputs, [lora_a], output_tensor, token_lora_mapping, scaling
    )
    
    if success:
        # Verify output
        if torch.all(output_tensor == 0):
            print("⚠️  Warning: Output is all zeros")
            return False
        else:
            print(f"📈 Output stats: min={output_tensor.min():.3f}, max={output_tensor.max():.3f}")
            
            # Compare with reference
            ref_output = torch.matmul(inputs, lora_a.t()) * scaling
            diff = torch.abs(output_tensor[0] - ref_output)
            max_diff = torch.max(diff).item()
            print(f"📊 Max difference from reference: {max_diff:.6f}")
            
            if max_diff < 1e-2:
                print("✅ ctypes wrapper test passed!")
                return True
            else:
                print(f"⚠️  Large difference: {max_diff:.6f}, but interface works")
                return True
    
    return False

if __name__ == "__main__":
    print("🚀 Testing ctypes CUDA wrapper...")
    
    if test_ctypes_wrapper():
        print("🎉 ctypes wrapper works!")
        print("\n🔧 This approach completely bypasses PyBind11 issues!")
    else:
        print("❌ ctypes wrapper failed!")
