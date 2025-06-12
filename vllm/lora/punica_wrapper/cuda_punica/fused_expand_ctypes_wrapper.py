"""
简化版QKV+LoRA融合expand操作的Python接口
模仿原始lora_expand_kernel的设计，外部提取好LoRA A数据再传入
"""

import ctypes
import os
from typing import Dict, Optional, Tuple, List

import torch

# 环境变量控制
VLLM_USE_FUSED_EXPAND = os.environ.get("VLLM_USE_FUSED_EXPAND", "1") == "1"

FUSED_EXPAND_AVAILABLE = False
C_LIB = None

if VLLM_USE_FUSED_EXPAND:
    try:
        _lib_path = os.path.join(os.path.dirname(__file__), "build", "libcuda_lora_c.so")
        if os.path.exists(_lib_path):
            C_LIB = ctypes.CDLL(_lib_path)
            FUSED_EXPAND_AVAILABLE = True
        else:
            print("❌ [vllm] Fused Expand Kernel: libcuda_lora_c.so not found.")
            
    except Exception as e:
        print(f"❌ [vllm] Failed to load CUDA LoRA C library: {e}")
else:
    print("ℹ️  [vllm] Fused Expand Kernel is disabled by environment variable.")

def _get_dtype_enum(tensor: torch.Tensor) -> int:
    """获取PyTorch张量对应的数据类型枚举值"""
    if tensor.dtype == torch.float16:
        return 0  # fp16
    elif tensor.dtype == torch.bfloat16:
        return 1  # bf16
    elif tensor.dtype == torch.float32:
        return 2  # fp32
    else:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")

def cuda_fused_qkv_expand_interface(
    fused_matmul_output: torch.Tensor,              # 融合矩阵乘法的完整输出
    output_tensor: torch.Tensor,                    # QKV输出张量
    lora_b_stacked: tuple,                          
    lora_bias_stacked: Optional[tuple],             
    output_slices: tuple,                           
    lora_a_slice_starts: torch.Tensor,              # 改为torch.Tensor
    lora_slice_ranks: torch.Tensor,                 # 改为torch.Tensor
    token_indices_sorted: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    qkv_output_size: int,                           # QKV部分的偏移量
    no_lora_flag: bool,
) -> bool:
    """
    调用简化版fused expand kernel
    
    Args:
        fused_matmul_output: 融合矩阵乘法的完整输出
        output_tensor: QKV输出张量
        lora_b_stacked: LoRA B权重元组
        lora_bias_stacked: LoRA bias元组
        output_slices: 输出slice大小
        lora_a_slice_starts: LoRA A slice起始位置
        lora_slice_ranks: 每个slice的rank大小
        ...其他参数与原始punica wrapper相同
    
    Returns:
        bool: 是否成功执行
    """
    if not FUSED_EXPAND_AVAILABLE:
        print("⚠️ [vllm] Fused expand not available")
        return False
    
    if no_lora_flag:
        print("ℹ️  [vllm] No LoRA active, skipping fused expand")
        return True
        
    try:
        # 基本参数
        max_active_loras = len(lora_b_stacked[0])
        num_slices = len(lora_b_stacked)
        M = fused_matmul_output.shape[0]
        max_hidden_size = max(output_slices)
        device = fused_matmul_output.device
        
        # 准备slice起始位置 (on GPU)
        slice_starts_list = []
        cumulative_size = 0
        for size in output_slices:
            slice_starts_list.append(cumulative_size)
            cumulative_size += size
        slice_starts = torch.tensor(slice_starts_list, dtype=torch.int, device=device)

        # 压平LoRA B权重中多余的维度 (num_loras, 1, hidden, rank) -> (num_loras, hidden, rank)
        lora_b_stacked_3d = [lora_b.squeeze(1) for lora_b in lora_b_stacked]

        # 构建LoRA B指针数组 (on GPU)
        lora_b_ptrs_list = [lora_b.data_ptr() for lora_b in lora_b_stacked_3d]
        lora_b_ptr_tensor = torch.tensor(lora_b_ptrs_list, dtype=torch.int64, device=device)
        
        # 构建其他元数据 (on GPU) - now for the 3D tensor
        hidden_sizes_list = [lora_b.shape[1] for lora_b in lora_b_stacked_3d]
        strides_d0_list = [lora_b.stride(0) for lora_b in lora_b_stacked_3d]
        strides_d1_list = [lora_b.stride(1) for lora_b in lora_b_stacked_3d]
        strides_d2_list = [lora_b.stride(2) for lora_b in lora_b_stacked_3d]

        hidden_sizes = torch.tensor(hidden_sizes_list, dtype=torch.int, device=device)
        lora_strides_d0 = torch.tensor(strides_d0_list, dtype=torch.int, device=device)
        lora_strides_d1 = torch.tensor(strides_d1_list, dtype=torch.int, device=device)
        lora_strides_d2 = torch.tensor(strides_d2_list, dtype=torch.int, device=device)
   
        # 调用C库函数
        result = C_LIB.cuda_lora_fused_expand_c(
            ctypes.c_void_p(fused_matmul_output.data_ptr()),
            ctypes.c_void_p(lora_b_ptr_tensor.data_ptr()),    
            ctypes.c_void_p(output_tensor.data_ptr()),         
            ctypes.c_void_p(token_indices_sorted.data_ptr()),  
            ctypes.c_void_p(lora_ids.data_ptr()),              
            ctypes.c_void_p(num_tokens_per_lora.data_ptr()),   
            ctypes.c_void_p(lora_token_start_loc.data_ptr()),  
            ctypes.c_void_p(slice_starts.data_ptr()),          # GPU pointer
            ctypes.c_void_p(lora_a_slice_starts.data_ptr()),   # GPU pointer
            ctypes.c_void_p(lora_slice_ranks.data_ptr()),      # GPU pointer
            ctypes.c_void_p(lora_strides_d0.data_ptr()),       # GPU pointer
            ctypes.c_void_p(lora_strides_d1.data_ptr()),       # GPU pointer
            ctypes.c_void_p(lora_strides_d2.data_ptr()),       # GPU pointer
            ctypes.c_void_p(hidden_sizes.data_ptr()),          # GPU pointer
            ctypes.c_int(max_active_loras),
            ctypes.c_int(M),                                   
            ctypes.c_int(max_hidden_size),                     
            ctypes.c_int(qkv_output_size),                     # qkv_output_size
            ctypes.c_int(num_slices),
            ctypes.c_bool(True),                               # add_inputs
            ctypes.c_int(fused_matmul_output.stride(0)),       # fused_input_stride0
            ctypes.c_int(fused_matmul_output.stride(1)),       # fused_input_stride1
            ctypes.c_int(output_tensor.stride(0)),             
            ctypes.c_int(output_tensor.stride(1)),             
            ctypes.c_void_p(0), # stream
            ctypes.c_int(_get_dtype_enum(fused_matmul_output)),# input_dtype
            ctypes.c_int(_get_dtype_enum(output_tensor)),      # output_dtype
        )

        if result == 0:
            # print("✅ [vllm] Fused expand completed successfully") # Comment out for cleaner logs
            return True
        else:
            print(f"❌ [vllm] Fused expand failed with return code: {result}")
            return False

    except Exception as e:
        print(f"❌ [vllm] Fused expand failed: {e}")
        return False


if __name__ == "__main__":
    if FUSED_EXPAND_AVAILABLE:
        # 为导出的C函数定义argtypes和restype，确保类型安全
        C_LIB.cuda_lora_fused_expand_c.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
        C_LIB.cuda_lora_fused_expand_c.restype = ctypes.c_int
    print(f"CUDA LoRA Fused Expand availability: {FUSED_EXPAND_AVAILABLE}")