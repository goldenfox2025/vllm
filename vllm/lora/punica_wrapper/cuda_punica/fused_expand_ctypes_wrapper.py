"""
Python ctypes wrapper for CUDA LoRA Fused Expand kernel
专门处理QKV+LoRA融合计算的expand操作的Python接口
"""

import ctypes
import torch
import numpy as np
from typing import List, Tuple, Optional

# 尝试加载CUDA LoRA库
try:
    from .ctypes_wrapper import cuda_lora_lib, C_LIB_AVAILABLE, _dtype_to_int
    FUSED_EXPAND_AVAILABLE = C_LIB_AVAILABLE
except ImportError:
    FUSED_EXPAND_AVAILABLE = False
    cuda_lora_lib = None

def cuda_lora_fused_expand_triton_interface(
    fused_shrink_input: torch.Tensor,     # [num_tokens, total_lora_rank]
    lora_b_weights: List[torch.Tensor],   # List of LoRA B weights for each slice
    output_tensor: torch.Tensor,         # [num_tokens, total_hidden_size]
    token_indices_sorted: torch.Tensor,   # Sorted token indices
    num_tokens_per_lora: torch.Tensor,    # Number of tokens per LoRA
    lora_token_start_loc: torch.Tensor,   # Start location for each LoRA
    lora_ids: torch.Tensor,               # Active LoRA IDs
    no_lora_flag: torch.Tensor,           # Flag indicating if LoRA is used
    slice_rank_info: List[dict],          # Each dict contains slice rank info
    offset_start: int = 0,
    add_inputs: bool = True
) -> bool:
    """
    CUDA LoRA Fused Expand kernel的Triton兼容接口
    
    Args:
        fused_shrink_input: 融合shrink结果 [num_tokens, total_lora_rank]
                           格式：total_lora_rank = max_loras * (slice0_rank + slice1_rank + slice2_rank)
        lora_b_weights: 每个slice的LoRA B权重列表
        output_tensor: 输出张量 [num_tokens, total_hidden_size]
        token_indices_sorted: 排序的token索引
        num_tokens_per_lora: 每个LoRA的token数量
        lora_token_start_loc: 每个LoRA的token起始位置
        lora_ids: 活跃LoRA ID
        no_lora_flag: 是否使用LoRA的标志
        slice_rank_info: 每个slice的rank信息列表（仅用于计算参数，kernel内部计算偏移）
        offset_start: 输出偏移起始位置
        add_inputs: 是否累加到输出
        
    Returns:
        bool: 成功返回True，失败返回False
    """
    
    if not FUSED_EXPAND_AVAILABLE:
        print("⚠️  CUDA LoRA fused expand kernel not available")
        return False
    
    if no_lora_flag.item():
        # 没有LoRA需要处理
        return True
    
    try:
        # 基本参数
        num_tokens = fused_shrink_input.size(0)
        total_lora_rank = fused_shrink_input.size(1)
        total_hidden_size = output_tensor.size(1)
        num_slices = len(lora_b_weights)
        max_active_loras = lora_ids.size(0)
        
        print(f"🔍 [Fused Expand] 参数分析:")
        print(f"   num_tokens: {num_tokens}")
        print(f"   total_lora_rank: {total_lora_rank}")
        print(f"   total_hidden_size: {total_hidden_size}")
        print(f"   num_slices: {num_slices}")
        print(f"   max_active_loras: {max_active_loras}")
        
        # 构建slice信息数组
        slice_starts = []
        slice_ranks = []
        hidden_sizes = []
        
        current_hidden_start = 0
        
        for i, info in enumerate(slice_rank_info):
            slice_idx = info['slice_idx']
            rank = info['rank']
            
            # 计算slice的hidden_size (从lora_b_weights推断)
            if i < len(lora_b_weights):
                hidden_size = lora_b_weights[i].size(1)  # [lora_id, hidden_size, rank]
            else:
                hidden_size = 0
            
            slice_starts.append(current_hidden_start)
            slice_ranks.append(rank)
            hidden_sizes.append(hidden_size)
            
            current_hidden_start += hidden_size
            
            print(f"   Slice {slice_idx}: rank={rank}, hidden_size={hidden_size}, start={slice_starts[-1]}")
        
        # 转换为CUDA张量
        slice_starts_tensor = torch.tensor(slice_starts, dtype=torch.int32, device=fused_shrink_input.device)
        slice_ranks_tensor = torch.tensor(slice_ranks, dtype=torch.int32, device=fused_shrink_input.device)
        hidden_sizes_tensor = torch.tensor(hidden_sizes, dtype=torch.int32, device=fused_shrink_input.device)
        
        # slice_rank_starts已经不需要了，但为了保持接口兼容性，创建一个dummy tensor
        slice_rank_starts_tensor = torch.zeros(num_slices, dtype=torch.int32, device=fused_shrink_input.device)
        
        # 构建LoRA B权重指针数组
        lora_b_ptr_array = torch.zeros(num_slices, dtype=torch.int64, device=fused_shrink_input.device)
        lora_strides_d0 = torch.zeros(num_slices, dtype=torch.int32, device=fused_shrink_input.device)
        lora_strides_d1 = torch.zeros(num_slices, dtype=torch.int32, device=fused_shrink_input.device)
        lora_strides_d2 = torch.zeros(num_slices, dtype=torch.int32, device=fused_shrink_input.device)
        
        for i, lora_b in enumerate(lora_b_weights):
            lora_b_ptr_array[i] = lora_b.data_ptr()
            lora_strides_d0[i] = lora_b.stride(0)  # LoRA ID维度
            lora_strides_d1[i] = lora_b.stride(1)  # hidden维度
            lora_strides_d2[i] = lora_b.stride(2)  # rank维度
            
            print(f"   LoRA B[{i}] shape: {lora_b.shape}, strides: ({lora_b.stride(0)}, {lora_b.stride(1)}, {lora_b.stride(2)})")
        
        # 数据类型转换
        input_dtype = _dtype_to_int(fused_shrink_input.dtype)
        output_dtype = _dtype_to_int(output_tensor.dtype)
        
        print(f"🚀 [Fused Expand] 调用CUDA kernel...")
        
        # 调用CUDA kernel
        result = cuda_lora_lib.cuda_lora_fused_expand_c(
            # 输入数据指针
            ctypes.c_void_p(fused_shrink_input.data_ptr()),
            ctypes.c_void_p(lora_b_ptr_array.data_ptr()),
            ctypes.c_void_p(output_tensor.data_ptr()),
            
            # 元数据指针
            ctypes.c_void_p(token_indices_sorted.data_ptr()),
            ctypes.c_void_p(lora_ids.data_ptr()),
            ctypes.c_void_p(num_tokens_per_lora.data_ptr()),
            ctypes.c_void_p(lora_token_start_loc.data_ptr()),
            
            # slice信息
            ctypes.c_void_p(slice_starts_tensor.data_ptr()),
            ctypes.c_void_p(slice_ranks_tensor.data_ptr()),
            ctypes.c_void_p(slice_rank_starts_tensor.data_ptr()),  # dummy参数
            
            # LoRA权重信息
            ctypes.c_void_p(lora_strides_d0.data_ptr()),
            ctypes.c_void_p(lora_strides_d1.data_ptr()),
            ctypes.c_void_p(lora_strides_d2.data_ptr()),
            ctypes.c_void_p(hidden_sizes_tensor.data_ptr()),
            
            # 基本参数
            ctypes.c_int(max_active_loras),
            ctypes.c_int(num_tokens),
            ctypes.c_int(total_hidden_size),
            ctypes.c_int(num_slices),
            ctypes.c_int(1 if add_inputs else 0),
            
            # stride信息
            ctypes.c_int(fused_shrink_input.stride(0)),
            ctypes.c_int(fused_shrink_input.stride(1)),
            ctypes.c_int(output_tensor.stride(0)),
            ctypes.c_int(output_tensor.stride(1)),
            
            # stream和数据类型
            ctypes.c_void_p(0),  # 使用默认流
            ctypes.c_int(input_dtype),
            ctypes.c_int(output_dtype)
        )
        
        if result != 0:
            print(f"❌ CUDA fused expand kernel failed with code: {result}")
            return False
        
        print("🚀 CUDA fused expand kernel completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ CUDA fused expand kernel error: {e}")
        return False


if __name__ == "__main__":
    # 测试代码
    print(f"CUDA LoRA Fused Expand availability: {FUSED_EXPAND_AVAILABLE}")
    if FUSED_EXPAND_AVAILABLE:
        print("✅ CUDA LoRA fused expand kernel is available for testing")
    else:
        print("❌ CUDA LoRA fused expand kernel is not available") 