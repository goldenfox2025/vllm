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
        print(f"Ultimate fusion CUDA library loaded: {lib_path}")
    else:
        print(f"C library not found at: {lib_path}")

except Exception as e:
    print(f"Failed to load C library for ultimate fusion: {e}")

def cuda_ultimate_fusion_interface(
    inputs: torch.Tensor,                    # [num_tokens, hidden_size]
    qkv_weights: torch.Tensor,               # [qkv_output_size, hidden_size]
    lora_a_stacked: tuple[torch.Tensor, ...], # tuple of LoRA A weights for each slice
    lora_b_stacked: tuple[torch.Tensor, ...], # tuple of LoRA B weights for each slice
    output_slices: tuple[int, ...],          # output size for each slice
    token_indices_sorted: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    lora_ranks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    终极融合内核的Python接口
    
    在一个CUDA内核中完成:
    1. 基础QKV计算: output_base = inputs @ qkv_weights.T
    2. LoRA路径计算: output_lora = (inputs @ lora_a) @ lora_b
    3. 结果融合: output = output_base + output_lora
    """
    device = inputs.device
    
    # 严格的设备检查 - 确保所有tensor都在GPU上并且是contiguous的
    all_tensors = [
        ("inputs", inputs),
        ("qkv_weights", qkv_weights),
        ("token_indices_sorted", token_indices_sorted),
        ("num_tokens_per_lora", num_tokens_per_lora),
        ("lora_token_start_loc", lora_token_start_loc),
        ("lora_ids", lora_ids),
    ]
    
    for name, tensor in all_tensors:
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on CUDA device, got {tensor.device}")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
            
    # 检查LoRA权重并去除额外的1维度 (关键修复！)
    print(f"🔧 修复前 - LoRA A shapes: {[lora_a.shape for lora_a in lora_a_stacked]}")
    print(f"🔧 修复前 - LoRA B shapes: {[lora_b.shape for lora_b in lora_b_stacked]}")
    
    # 处理LoRA A权重：去除1维度
    lora_a_processed = []
    for i, lora_a in enumerate(lora_a_stacked):
        if not lora_a.is_cuda:
            raise ValueError(f"lora_a[{i}] must be on CUDA device, got {lora_a.device}")
        if not lora_a.is_contiguous():
            lora_a = lora_a.contiguous()
            
        # 去除额外的1维度，就像传统方法一样
        if lora_a.ndim == 4:  # shape:(lora_num,1,rank,hidden_size)
            assert lora_a.size(1) == 1, f"Expected size(1)==1, got {lora_a.shape}"
            lora_a = lora_a.squeeze(dim=1)  # -> (lora_num,rank,hidden_size)
        else:
            assert lora_a.ndim == 3, f"Expected 3D tensor after squeeze, got {lora_a.shape}"
        lora_a_processed.append(lora_a)
    
    # 处理LoRA B权重：去除1维度
    lora_b_processed = []
    for i, lora_b in enumerate(lora_b_stacked):
        if not lora_b.is_cuda:
            raise ValueError(f"lora_b[{i}] must be on CUDA device, got {lora_b.device}")
        if not lora_b.is_contiguous():
            lora_b = lora_b.contiguous()
            
        # 去除额外的1维度，就像传统方法一样
        if lora_b.ndim == 4:  # shape:(lora_num,1,output_size,rank)
            assert lora_b.size(1) == 1, f"Expected size(1)==1, got {lora_b.shape}"
            lora_b = lora_b.squeeze(dim=1)  # -> (lora_num,output_size,rank)
        else:
            assert lora_b.ndim == 3, f"Expected 3D tensor after squeeze, got {lora_b.shape}"
        lora_b_processed.append(lora_b)
    
    print(f"🔧 修复后 - LoRA A shapes: {[lora_a.shape for lora_a in lora_a_processed]}")
    print(f"🔧 修复后 - LoRA B shapes: {[lora_b.shape for lora_b in lora_b_processed]}")
    
    # 基本参数
    num_tokens = inputs.shape[0]
    hidden_size = inputs.shape[1]
    qkv_output_size = qkv_weights.shape[0]
    num_slices = len(lora_a_processed)
    
    # 创建输出tensor
    output = torch.zeros(num_tokens, qkv_output_size, dtype=inputs.dtype, device=device)
    
    # 准备slice起始位置 (确保在GPU上)
    slice_starts_list = []
    cumulative_size = 0
    for size in output_slices:
        slice_starts_list.append(cumulative_size)
        cumulative_size += size
    slice_starts = torch.tensor(slice_starts_list, dtype=torch.int32, device=device)
    
    # 准备LoRA A指针数组 (确保在GPU上)
    lora_a_ptrs = []
    for lora_a in lora_a_processed:
        lora_a_ptrs.append(lora_a.data_ptr())
    lora_a_ptr_tensor = torch.tensor(lora_a_ptrs, dtype=torch.int64, device=device)
    
    # 准备LoRA B指针数组 (确保在GPU上)
    lora_b_ptrs = []
    for lora_b in lora_b_processed:
        lora_b_ptrs.append(lora_b.data_ptr())
    lora_b_ptr_tensor = torch.tensor(lora_b_ptrs, dtype=torch.int64, device=device)
    
    # 计算max_rank
    max_rank = max(lora_a.shape[1] for lora_a in lora_a_processed)
    
    # 准备ranks数组 (确保在GPU上)
    if lora_ranks is None:
        ranks_list = []
        max_loras = len(lora_ids)
        for i in range(max_loras):
            if i < len(lora_ids) and lora_ids[i] != -1:
                # 简化：假设所有LoRA都有相同的rank
                ranks_list.append(max_rank)
            else:
                ranks_list.append(0)
        lora_ranks = torch.tensor(ranks_list, dtype=torch.int32, device=device)
    else:
        if not lora_ranks.is_cuda:
            lora_ranks = lora_ranks.to(device)
        if not lora_ranks.is_contiguous():
            lora_ranks = lora_ranks.contiguous()
    
    # 计算strides (使用处理后的3D tensor)
    input_stride0 = inputs.stride(0)
    input_stride1 = inputs.stride(1)
    qkv_stride0 = qkv_weights.stride(0)
    qkv_stride1 = qkv_weights.stride(1)
    output_stride0 = output.stride(0)
    output_stride1 = output.stride(1)
    
    # LoRA strides (使用处理后的3D tensor，假设所有slice有相同的stride模式)
    lora_a_stride0 = lora_a_processed[0].stride(0)  # lora_id维度
    lora_a_stride1 = lora_a_processed[0].stride(1)  # rank维度  
    lora_a_stride2 = lora_a_processed[0].stride(2)  # hidden维度
    
    lora_b_stride0 = lora_b_processed[0].stride(0)  # lora_id维度
    lora_b_stride1 = lora_b_processed[0].stride(1)  # output维度
    lora_b_stride2 = lora_b_processed[0].stride(2)  # rank维度
    
    print(f"🔧 使用的LoRA A strides: [{lora_a_stride0}, {lora_a_stride1}, {lora_a_stride2}]")
    print(f"🔧 使用的LoRA B strides: [{lora_b_stride0}, {lora_b_stride1}, {lora_b_stride2}]")
    
    # 数据类型映射
    dtype_map = {
        torch.float16: 0,
        torch.bfloat16: 1, 
        torch.float32: 2
    }
    
    input_dtype = dtype_map.get(inputs.dtype)
    output_dtype = dtype_map.get(output.dtype)
    
    if input_dtype is None or output_dtype is None:
        raise ValueError(f"Unsupported dtype: input={inputs.dtype}, output={output.dtype}")
    
    print(f"🔧 调用终极融合内核...")
    print(f"   num_tokens={num_tokens}, hidden_size={hidden_size}")
    print(f"   qkv_output_size={qkv_output_size}, num_slices={num_slices}")
    print(f"   max_rank={max_rank}, input_dtype={input_dtype}, output_dtype={output_dtype}")
    
    # 调用CUDA内核
    result = cuda_c_lib.cuda_ultimate_fusion_c(
        inputs.data_ptr(),
        qkv_weights.data_ptr(),
        lora_a_ptr_tensor.data_ptr(),
        lora_b_ptr_tensor.data_ptr(),
        output.data_ptr(),
        token_indices_sorted.data_ptr(),
        lora_ids.data_ptr(),
        num_tokens_per_lora.data_ptr(),
        lora_token_start_loc.data_ptr(),
        slice_starts.data_ptr(),
        lora_ranks.data_ptr(),
        len(lora_ids),  # max_active_loras
        num_tokens,
        hidden_size,
        qkv_output_size,
        num_slices,
        max_rank,
        # Input strides
        input_stride0,
        input_stride1,
        # QKV weight strides
        qkv_stride0,
        qkv_stride1,
        # LoRA A strides (修复后的3D tensor strides)
        lora_a_stride0,
        lora_a_stride1,
        lora_a_stride2,
        # LoRA B strides (修复后的3D tensor strides)
        lora_b_stride0,
        lora_b_stride1,
        lora_b_stride2,
        # Output strides
        output_stride0,
        output_stride1,
        # Stream (null for default stream)
        0,
        # Data types
        input_dtype,
        output_dtype
    )
    
    if result != 0:
        raise RuntimeError(f"Ultimate fusion kernel failed with code: {result}")
    
    print(f"✅ 终极融合内核成功完成!")
    return output

def test_ultimate_fusion():
    """测试终极融合内核"""
    if not C_LIB_AVAILABLE:
        print("❌ Ultimate fusion library not available")
        return False
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print("🧪 Testing ultimate fusion kernel...")
    
    # 测试配置
    num_tokens = 4
    hidden_size = 8
    qkv_output_size = 12  # Q(4) + K(4) + V(4)
    rank = 2
    num_slices = 3  # Q, K, V
    
    device = torch.device('cuda:0')
    dtype = torch.float16
    
    # 创建测试张量
    inputs = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    qkv_weights = torch.randn(qkv_output_size, hidden_size, dtype=dtype, device=device)
    
    # 创建LoRA权重 (每个slice一个)
    lora_a_stacked = tuple(
        torch.randn(1, rank, hidden_size, dtype=dtype, device=device)  # [max_loras, rank, hidden]
        for _ in range(num_slices)
    )
    lora_b_stacked = tuple(
        torch.randn(1, 4, rank, dtype=dtype, device=device)  # [max_loras, slice_output, rank]
        for _ in range(num_slices)
    )
    
    output_slices = (4, 4, 4)  # Q, K, V各4维
    
    # 创建简单的Punica元数据（所有token使用同一个LoRA）
    token_indices_sorted = torch.arange(num_tokens, dtype=torch.int32, device=device)
    num_tokens_per_lora = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    lora_token_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    lora_ids = torch.tensor([0], dtype=torch.int32, device=device)
    
    try:
        # 调用终极融合内核
        output = cuda_ultimate_fusion_interface(
            inputs, qkv_weights, lora_a_stacked, lora_b_stacked, output_slices,
            token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, lora_ids
        )
        
        print(f"✅ Ultimate fusion kernel test passed!")
        print(f"📊 Output shape: {output.shape}")
        print(f"📈 Output stats: min={output.min():.3f}, max={output.max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ultimate fusion kernel test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing ultimate fusion ctypes wrapper...")
    
    if test_ultimate_fusion():
        print("🎉 Ultimate fusion wrapper works!")
        print("\n🔧 This is the ULTIMATE optimization - everything in one kernel!")
    else:
        print("❌ Ultimate fusion wrapper failed!") 