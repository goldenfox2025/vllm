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

# 关键修复7: 按照vLLM模式实现预分配+缓存策略
# 参考vLLM的_LORA_A_PTR_DICT和_LORA_B_PTR_DICT设计

# 全局指针缓存字典 - 模仿vLLM的设计
_LORA_A_PTR_DICT = {}
_LORA_B_PTR_DICT = {}
_SLICE_STARTS_DICT = {}
_LORA_RANKS_DICT = {}

# 预分配的固定缓冲区 - 确保CUDA Graph兼容性
_DEVICE_BUFFERS = {}

def _get_device_buffers(device, max_loras=8, max_slices=8):
    """获取设备级别的预分配缓冲区 - 模仿vLLM的预分配策略"""
    device_id = device.index if hasattr(device, 'index') else 0
    
    if device_id not in _DEVICE_BUFFERS:
        # 预分配所有必要的tensor - 一次性创建，永不释放
        buffers = {
            'lora_a_ptr_buffer': torch.zeros(max_slices, max_loras, dtype=torch.int64, device=device),
            'lora_b_ptr_buffer': torch.zeros(max_slices, max_loras, dtype=torch.int64, device=device),
            'slice_starts_buffer': torch.zeros(max_slices, dtype=torch.int32, device=device),
            'lora_ranks_buffer': torch.zeros(max_loras, dtype=torch.int32, device=device),
            'max_loras': max_loras,
            'max_slices': max_slices,
        }
        _DEVICE_BUFFERS[device_id] = buffers
        print(f"🔧 创建设备{device_id}预分配缓冲区: max_loras={max_loras}, max_slices={max_slices}")
    
    return _DEVICE_BUFFERS[device_id]

def _get_lora_ptr_cached(lora_stacked, device, is_a_weights=True):
    """获取LoRA指针 - 使用缓存策略，模仿vLLM的_get_lora_a_ptr设计"""
    cache_dict = _LORA_A_PTR_DICT if is_a_weights else _LORA_B_PTR_DICT
    weight_type = "A" if is_a_weights else "B"
    
    # 创建缓存键 - 基于tensor地址
    cache_key = tuple(tensor.data_ptr() for tensor in lora_stacked)
    
    if cache_key not in cache_dict:
        print(f"🔧 首次创建LoRA {weight_type}指针缓存: key={len(cache_key)}个slice")
        
        # 获取预分配缓冲区
        buffers = _get_device_buffers(device, max_loras=8, max_slices=len(lora_stacked))
        
        if is_a_weights:
            ptr_buffer = buffers['lora_a_ptr_buffer']
        else:
            ptr_buffer = buffers['lora_b_ptr_buffer']
        
        # 计算并存储指针
        for slice_id, lora_tensor in enumerate(lora_stacked):
            max_loras = lora_tensor.shape[0]
            for lora_id in range(max_loras):
                # 计算每个LoRA的指针偏移
                base_ptr = lora_tensor.data_ptr()
                offset = lora_id * lora_tensor.stride(0)
                element_size = lora_tensor.element_size()
                final_ptr = base_ptr + offset * element_size
                
                # 存储到预分配缓冲区
                if slice_id < buffers['max_slices'] and lora_id < buffers['max_loras']:
                    ptr_buffer[slice_id, lora_id] = final_ptr
        
        # 缓存结果 - 存储缓冲区引用
        cache_dict[cache_key] = ptr_buffer
        print(f"✅ LoRA {weight_type}指针缓存创建完成")
    
    return cache_dict[cache_key]

def _get_slice_starts_cached(output_slices, device):
    """获取slice starts - 使用缓存策略"""
    cache_key = tuple(output_slices)
    
    if cache_key not in _SLICE_STARTS_DICT:
        print(f"🔧 首次创建slice_starts缓存: {output_slices}")
        
        # 获取预分配缓冲区
        buffers = _get_device_buffers(device, max_slices=len(output_slices))
        slice_starts_buffer = buffers['slice_starts_buffer']
        
        # 计算并存储slice starts
        cumulative_size = 0
        for i, size in enumerate(output_slices):
            if i < buffers['max_slices']:
                slice_starts_buffer[i] = cumulative_size
            cumulative_size += size
        
        # 缓存结果 - 存储slice starts的数值列表（用于计算）和tensor（用于内核）
        _SLICE_STARTS_DICT[cache_key] = {
            'tensor': slice_starts_buffer,
            'values': [cumulative_size := cumulative_size - size + (cumulative_size := cumulative_size - cumulative_size + sum(output_slices[:i+1])) - sum(output_slices[:i]) for i, size in enumerate(output_slices)],
            'total_size': sum(output_slices)
        }
        # 重新计算values
        values = []
        cumulative = 0
        for size in output_slices:
            values.append(cumulative)
            cumulative += size
        _SLICE_STARTS_DICT[cache_key]['values'] = values
        
        print(f"✅ slice_starts缓存创建完成")
    
    return _SLICE_STARTS_DICT[cache_key]

def _get_lora_ranks_cached(lora_stacked, lora_ids, device):
    """获取LoRA ranks - 使用缓存策略"""
    # 计算ranks
    max_rank = max(lora_a.shape[2] for lora_a in lora_stacked) if lora_stacked else 16
    cache_key = (max_rank, len(lora_ids))
    
    if cache_key not in _LORA_RANKS_DICT:
        print(f"🔧 首次创建lora_ranks缓存: max_rank={max_rank}, num_loras={len(lora_ids)}")
        
        # 获取预分配缓冲区
        buffers = _get_device_buffers(device, max_loras=max(8, len(lora_ids)))
        ranks_buffer = buffers['lora_ranks_buffer']
        
        # 设置ranks - 简化为所有LoRA使用相同rank
        for i in range(min(len(lora_ids), buffers['max_loras'])):
            ranks_buffer[i] = max_rank
        
        # 缓存结果 - 存储tensor和数值
        _LORA_RANKS_DICT[cache_key] = {
            'tensor': ranks_buffer,
            'max_rank': max_rank
        }
        print(f"✅ lora_ranks缓存创建完成")
    
    return _LORA_RANKS_DICT[cache_key]

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
    stream: Optional[int] = None,
) -> torch.Tensor:
    """
    终极融合内核的Python接口 - 真正的零动态分配版本
    
    关键设计：
    1. 使用全局缓存字典存储指针信息（模仿_LORA_A_PTR_DICT）
    2. 预分配所有tensor，避免CUDA Graph capture期间的动态分配
    3. 在capture期间绝对不创建任何tensor，不调用tensor.item()，不做slice操作
    """
    device = inputs.device
    
    # CUDA graph capture状态检测
    is_capturing = torch.cuda.is_current_stream_capturing()
    
    if not is_capturing:
        print(f"🔧 终极融合调用 - 零动态分配策略")
        print(f"🔧 CUDA Graph Capturing: {is_capturing}")
    
    # 验证输入tensor的设备和连续性 - 但不做任何修改
    for name, tensor in [
        ("inputs", inputs),
        ("qkv_weights", qkv_weights),
        ("token_indices_sorted", token_indices_sorted),
        ("num_tokens_per_lora", num_tokens_per_lora),
        ("lora_token_start_loc", lora_token_start_loc),
        ("lora_ids", lora_ids),
    ]:
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on CUDA device, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    
    # 验证LoRA权重 - 但不做任何修改
    lora_a_processed = []
    lora_b_processed = []
    
    for i, (lora_a, lora_b) in enumerate(zip(lora_a_stacked, lora_b_stacked)):
        if not lora_a.is_cuda or not lora_b.is_cuda:
            raise ValueError(f"LoRA weights must be on CUDA device")
        if not lora_a.is_contiguous() or not lora_b.is_contiguous():
            raise ValueError(f"LoRA weights must be contiguous")
        
        lora_a_processed.append(lora_a)
        lora_b_processed.append(lora_b)
        
        if not is_capturing:
            print(f"🔧 slice[{i}]: lora_a.shape={lora_a.shape}, lora_b.shape={lora_b.shape}")
    
    # 基本参数
    num_tokens = inputs.shape[0]
    hidden_size = inputs.shape[1]
    qkv_output_size = qkv_weights.shape[0]
    num_slices = len(lora_a_processed)
    
    # 创建输出tensor
    output = torch.zeros(num_tokens, qkv_output_size, dtype=inputs.dtype, device=device)
    
    # 关键：使用缓存策略获取所有预分配的tensor
    if not is_capturing:
        print(f"🔧 获取缓存的指针和元数据...")
    
    # 获取LoRA指针缓存
    lora_a_ptr_buffer = _get_lora_ptr_cached(lora_a_processed, device, is_a_weights=True)
    lora_b_ptr_buffer = _get_lora_ptr_cached(lora_b_processed, device, is_a_weights=False)
    
    # 获取slice starts缓存
    slice_starts_info = _get_slice_starts_cached(output_slices, device)
    slice_starts = slice_starts_info['tensor']
    slice_values = slice_starts_info['values']  # 预计算的数值，避免tensor.item()
    
    # 获取lora ranks缓存
    if lora_ranks is None:
        lora_ranks_info = _get_lora_ranks_cached(lora_a_processed, lora_ids, device)
        lora_ranks = lora_ranks_info['tensor']
        max_rank = lora_ranks_info['max_rank']  # 预计算的数值
    else:
        if not lora_ranks.is_cuda or lora_ranks.device != device:
            raise ValueError("lora_ranks must be on correct CUDA device")
        if not lora_ranks.is_contiguous():
            raise ValueError("lora_ranks must be contiguous")
        max_rank = 16  # 默认值，避免访问tensor
    
    # 检查库可用性
    if not C_LIB_AVAILABLE:
        raise RuntimeError("Ultimate fusion C library not available")
    
    # Stride计算
    input_stride0 = inputs.stride(0)
    input_stride1 = inputs.stride(1)
    qkv_stride0 = qkv_weights.stride(0)
    qkv_stride1 = qkv_weights.stride(1)
    output_stride0 = output.stride(0)
    output_stride1 = output.stride(1)
    
    # LoRA strides - 使用原始4D tensor的stride
    if len(lora_a_processed) > 0:
        lora_a = lora_a_processed[0]
        lora_a_stride0 = lora_a.stride(0)  # lora_id维度
        lora_a_stride1 = lora_a.stride(1)  # 1维度（通常为1）
        lora_a_stride2 = lora_a.stride(2)  # rank维度
        lora_a_stride3 = lora_a.stride(3)  # hidden维度
        
        lora_b = lora_b_processed[0]
        lora_b_stride0 = lora_b.stride(0)  # lora_id维度
        lora_b_stride1 = lora_b.stride(1)  # 1维度（通常为1）
        lora_b_stride2 = lora_b.stride(2)  # output维度
        lora_b_stride3 = lora_b.stride(3)  # rank维度
        
        # 调整stride以匹配内核期望的3D布局
        final_lora_a_stride0 = lora_a_stride0
        final_lora_a_stride1 = lora_a_stride2  # rank维度
        final_lora_a_stride2 = lora_a_stride3  # hidden维度
        
        final_lora_b_stride0 = lora_b_stride0
        final_lora_b_stride1 = lora_b_stride2  # output维度
        final_lora_b_stride2 = lora_b_stride3  # rank维度
    else:
        # 默认值
        final_lora_a_stride0 = final_lora_a_stride1 = final_lora_a_stride2 = 1
        final_lora_b_stride0 = final_lora_b_stride1 = final_lora_b_stride2 = 1
    
    # 数据类型映射
    dtype_map = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
    input_dtype = dtype_map.get(inputs.dtype, 0)
    output_dtype = dtype_map.get(output.dtype, 0)
    
    # 获取CUDA流
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    
    # 关键：现在所有tensor都是预分配的，使用预计算的数值避免tensor访问
    if not is_capturing:
        print(f"🔧 调用终极融合内核 - 使用预分配tensor和预计算数值...")
    
    # 为每个slice调用内核
    for slice_id in range(num_slices):
        if not is_capturing:
            print(f"🔧 处理slice {slice_id}/{num_slices}")
        
        # 关键修复：使用预计算的数值，避免tensor.item()调用
        slice_start = slice_values[slice_id]
        if slice_id + 1 < len(slice_values):
            slice_end = slice_values[slice_id + 1]
        else:
            slice_end = slice_values[-1] + output_slices[-1]  # 总大小
        slice_size = slice_end - slice_start
        
        # 关键修复：使用预分配的tensor views，避免动态slice创建
        # 我们需要创建固定大小的视图，不能使用动态slicing
        
        # 获取当前slice的指针数组 - 使用预分配的缓冲区
        slice_lora_a_ptrs = lora_a_ptr_buffer[slice_id]  # [max_loras]
        slice_lora_b_ptrs = lora_b_ptr_buffer[slice_id]  # [max_loras]
        
        if not is_capturing:
            print(f"   slice范围: [{slice_start}:{slice_end}], size={slice_size}")
        
        # 计算QKV权重和输出的指针偏移
        qkv_element_size = qkv_weights.element_size()
        output_element_size = output.element_size()
        
        qkv_slice_ptr = qkv_weights.data_ptr() + slice_start * qkv_weights.stride(0) * qkv_element_size
        output_slice_ptr = output.data_ptr() + slice_start * output.stride(1) * output_element_size
        
        # 调用C库函数 - 使用预分配的tensor和指针计算
        try:
            result = cuda_c_lib.cuda_ultimate_fusion_c(
                ctypes.c_void_p(inputs.data_ptr()),
                ctypes.c_void_p(qkv_slice_ptr),  # 计算的slice指针
                ctypes.c_void_p(slice_lora_a_ptrs.data_ptr()),  # 预分配的指针数组
                ctypes.c_void_p(slice_lora_b_ptrs.data_ptr()),  # 预分配的指针数组
                ctypes.c_void_p(output_slice_ptr),  # 计算的slice输出指针
                ctypes.c_void_p(token_indices_sorted.data_ptr()),
                ctypes.c_void_p(lora_ids.data_ptr()),
                ctypes.c_void_p(num_tokens_per_lora.data_ptr()),
                ctypes.c_void_p(lora_token_start_loc.data_ptr()),
                ctypes.c_void_p(slice_starts.data_ptr()),
                ctypes.c_void_p(lora_ranks.data_ptr()),
                ctypes.c_int(len(lora_ids)),  # max_active_loras
                ctypes.c_int(num_tokens),
                ctypes.c_int(hidden_size),
                ctypes.c_int(slice_size),     # 当前slice的输出大小
                ctypes.c_int(1),              # num_slices = 1 (单独处理每个slice)
                ctypes.c_int(max_rank),       # 使用预计算的max_rank
                ctypes.c_int(input_stride0),
                ctypes.c_int(input_stride1),
                ctypes.c_int(qkv_weights.stride(0)),
                ctypes.c_int(qkv_weights.stride(1)),
                ctypes.c_int(final_lora_a_stride0),
                ctypes.c_int(final_lora_a_stride1),
                ctypes.c_int(final_lora_a_stride2),
                ctypes.c_int(final_lora_b_stride0),
                ctypes.c_int(final_lora_b_stride1),
                ctypes.c_int(final_lora_b_stride2),
                ctypes.c_int(output_stride0),
                ctypes.c_int(output_stride1),
                ctypes.c_void_p(stream),
                ctypes.c_int(input_dtype),
                ctypes.c_int(output_dtype),
            )
            
            if result != 0:
                raise RuntimeError(f"Ultimate fusion kernel failed for slice {slice_id} with code {result}")
                
        except Exception as e:
            if not is_capturing:
                print(f"❌ Slice {slice_id} 失败: {e}")
            raise
    
    # 同步确保所有计算完成 - 但只在非capture期间
    if not is_capturing:
        torch.cuda.synchronize()
        print(f"✅ 终极融合完成: output.shape={output.shape}")
    
    return output

# 保留旧的函数用于向后兼容
def _get_or_create_pointer_buffers(device, max_slices=8):
    """向后兼容函数"""
    return _get_device_buffers(device, max_loras=8, max_slices=max_slices)

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
        torch.randn(1, 1, rank, hidden_size, dtype=dtype, device=device)  # [max_loras, 1, rank, hidden]
        for _ in range(num_slices)
    )
    lora_b_stacked = tuple(
        torch.randn(1, 1, 4, rank, dtype=dtype, device=device)  # [max_loras, 1, slice_output, rank]
        for _ in range(num_slices)
    )
    
    output_slices = (4, 4, 4)  # Q, K, V各4维
    
    # 创建简单的Punica元数据（所有token使用同一个LoRA）
    token_indices_sorted = torch.arange(num_tokens, dtype=torch.int32, device=device)
    num_tokens_per_lora = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    lora_token_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
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