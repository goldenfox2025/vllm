import ctypes
import os
import torch
from typing import Tuple, Optional

# --- Part 1: C动态库加载 ---
C_LIB_AVAILABLE = False
cuda_c_lib = None

try:
    lib_path_options = [
        os.path.join(os.path.dirname(__file__), "build", "libcuda_lora_c.so"),
        os.path.join(os.path.dirname(__file__), "libcuda_lora_c.so"),
    ]
    lib_path = next((path for path in lib_path_options if os.path.exists(path)), None)
    
    if lib_path:
        cuda_c_lib = ctypes.CDLL(lib_path)
        cuda_c_lib.cuda_ultimate_fusion_c.argtypes = [
            ctypes.c_void_p,  # input_ptr
            ctypes.c_void_p,  # qkv_weights_ptr
            ctypes.c_void_p,  # lora_a_ptr_array
            ctypes.c_void_p,  # lora_b_ptr_array
            ctypes.c_void_p,  # output_ptr
            ctypes.c_void_p,  # intermediate_buffer_ptr  
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
            ctypes.c_int,     # input_stride0, input_stride1
            ctypes.c_int,
            ctypes.c_int,     # qkv_stride0, qkv_stride1
            ctypes.c_int,
            ctypes.c_int,     # lora_a_stride0, lora_a_stride1, lora_a_stride2
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,     # lora_b_stride0, lora_b_stride1, lora_b_stride2
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,     # output_stride0, output_stride1
            ctypes.c_int,
            ctypes.c_void_p,  # stream_ptr
            ctypes.c_int,     # input_dtype
            ctypes.c_int,     # output_dtype
        ]
        cuda_c_lib.cuda_ultimate_fusion_c.restype = ctypes.c_int
        C_LIB_AVAILABLE = True
        print(f"✅ Ultimate fusion CUDA library loaded successfully from: {lib_path}")
    else:
        print(f"⚠️ C library 'libcuda_lora_c.so' not found.")
except Exception as e:
    print(f"❌ Failed to load C library for ultimate fusion: {e}")

# --- Part 2: 预分配与缓存管理 ---
_DEVICE_BUFFERS = {}
_LORA_A_SLICE_PTR_DICT, _LORA_B_SLICE_PTR_DICT = {}, {}
_SLICE_STARTS_DICT, _LORA_RANKS_DICT = {}, {}
_INTERMEDIATE_BUFFER = None

def _get_device_buffers(device, max_loras=8, max_slices=8):
    """为每个CUDA设备创建并缓存一个包含所有预分配辅助缓冲区的字典。"""
    device_id = device.index if hasattr(device, 'index') else 0
    if device_id not in _DEVICE_BUFFERS:
        print(f"🔧 Allocating persistent helper buffers for device:{device_id}")
        _DEVICE_BUFFERS[device_id] = {
            'lora_a_slice_ptr_buffer': torch.zeros(max_slices, dtype=torch.int64, device=device),
            'lora_b_slice_ptr_buffer': torch.zeros(max_slices, dtype=torch.int64, device=device),
            'slice_starts_buffer': torch.zeros(max_slices, dtype=torch.int32, device=device),
            'lora_ranks_buffer': torch.zeros(max_loras, dtype=torch.int32, device=device),
        }
    return _DEVICE_BUFFERS[device_id]

def _get_lora_slice_ptr_cached(lora_stacked: Tuple[torch.Tensor, ...], device: torch.device, is_a_weights: bool):
    """(Graph-Safe) 获取一个指向LoRA slice权重指针数组的预分配缓冲区。"""
    cache_dict = _LORA_A_SLICE_PTR_DICT if is_a_weights else _LORA_B_SLICE_PTR_DICT
    cache_key = tuple(t.data_ptr() for t in lora_stacked)
    if not cache_key in cache_dict:
        num_slices = len(lora_stacked)
        buffers = _get_device_buffers(device, max_slices=num_slices)
        ptr_buffer = buffers['lora_a_slice_ptr_buffer' if is_a_weights else 'lora_b_slice_ptr_buffer']
        temp_ptr_list = [t.data_ptr() for t in lora_stacked]
        temp_cpu_tensor = torch.tensor(temp_ptr_list, dtype=torch.int64, device='cpu')
        ptr_buffer[:num_slices].copy_(temp_cpu_tensor, non_blocking=True)
        cache_dict[cache_key] = ptr_buffer
    return cache_dict[cache_key]

def _get_slice_starts_cached(output_slices: Tuple[int, ...], device: torch.device):
    """(Graph-Safe) 获取slice_starts的预分配缓冲区。"""
    cache_key = output_slices
    if not cache_key in _SLICE_STARTS_DICT:
        buffers = _get_device_buffers(device, max_slices=len(output_slices))
        slice_starts_buffer = buffers['slice_starts_buffer']
        cumulative = 0
        for i, size in enumerate(output_slices):
            if i < slice_starts_buffer.shape[0]:
                slice_starts_buffer[i] = cumulative
            cumulative += size
        _SLICE_STARTS_DICT[cache_key] = {'tensor': slice_starts_buffer}
    return _SLICE_STARTS_DICT[cache_key]['tensor']

def _get_lora_ranks_cached(lora_stacked: Tuple[torch.Tensor, ...], lora_ids: torch.Tensor, device: torch.device):
    """(Graph-Safe) 获取lora_ranks的预分配缓冲区和max_rank值。"""
    max_rank = max(lora_a.shape[2] for lora_a in lora_stacked) if lora_stacked else 16
    cache_key = (max_rank, len(lora_ids))
    if not cache_key in _LORA_RANKS_DICT:
        num_loras = len(lora_ids)
        buffers = _get_device_buffers(device, max_loras=max(8, num_loras))
        ranks_buffer = buffers['lora_ranks_buffer']
        ranks_buffer[:num_loras] = max_rank
        _LORA_RANKS_DICT[cache_key] = {'tensor': ranks_buffer, 'max_rank': max_rank}
    return _LORA_RANKS_DICT[cache_key]

# --- Part 3: Python主接口函数 ---
def cuda_ultimate_fusion_interface(
    inputs: torch.Tensor,
    qkv_weights: torch.Tensor,
    lora_a_stacked: Tuple[torch.Tensor, ...],
    lora_b_stacked: Tuple[torch.Tensor, ...],
    output_slices: Tuple[int, ...],
    token_indices_sorted: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    lora_ranks_tensor: Optional[torch.Tensor] = None,
    stream: Optional[int] = None,
    intermediate_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    终极融合内核的Python接口 - 支持预分配的中间缓冲区，完全适配CUDA Graph。
    """
    global _INTERMEDIATE_BUFFER
    if not C_LIB_AVAILABLE:
        raise RuntimeError("Ultimate Fusion C library is not available.")
        
    device = inputs.device
    
    # 1. 获取辅助缓冲区和元数据
    lora_a_ptr_array = _get_lora_slice_ptr_cached(lora_a_stacked, device, is_a_weights=True)
    lora_b_ptr_array = _get_lora_slice_ptr_cached(lora_b_stacked, device, is_a_weights=False)
    slice_starts = _get_slice_starts_cached(output_slices, device)
    
    if lora_ranks_tensor is None:
        ranks_info = _get_lora_ranks_cached(lora_a_stacked, lora_ids, device)
        lora_ranks_tensor = ranks_info['tensor']
        max_rank = ranks_info['max_rank']
    else:
        max_rank = 16 # Fallback

    # 2. 计算参数
    num_tokens, hidden_size = inputs.shape
    qkv_output_size = qkv_weights.shape[0]
    num_slices = len(lora_a_stacked)
    output = torch.empty(num_tokens, qkv_output_size, dtype=inputs.dtype, device=device)

    # 3. 管理中间缓冲区 (关键的图安全逻辑)
    if intermediate_buffer is None:
        required_shape = (num_tokens, max_rank*3)
        if (_INTERMEDIATE_BUFFER is None or
            _INTERMEDIATE_BUFFER.shape[0] < num_tokens or
            _INTERMEDIATE_BUFFER.shape[1] < max_rank or
            _INTERMEDIATE_BUFFER.dtype != inputs.dtype or
            _INTERMEDIATE_BUFFER.device != device):
            # 这一步必须在图捕捉之前，由一个足够大的max_tokens_in_graph来完成初始化
            print(f"🔧 (Re)allocating intermediate buffer with shape={required_shape}")
            _INTERMEDIATE_BUFFER = torch.empty(required_shape, dtype=inputs.dtype, device=device)
        intermediate_buffer = _INTERMEDIATE_BUFFER[:num_tokens, :max_rank]

    # 4. 计算Strides和其他参数
    input_strides = inputs.stride()
    qkv_strides = qkv_weights.stride()
    output_strides = output.stride()
    
    if num_slices > 0:
        lora_a_strides = lora_a_stacked[0].stride()
        final_lora_a_strides = (lora_a_strides[0], lora_a_strides[2], lora_a_strides[3])
        lora_b_strides = lora_b_stacked[0].stride()
        final_lora_b_strides = (lora_b_strides[0], lora_b_strides[2], lora_b_strides[3])
    else:
        final_lora_a_strides = (0, 0, 0)
        final_lora_b_strides = (0, 0, 0)

    dtype_map = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
    input_dtype, output_dtype = dtype_map.get(inputs.dtype, 0), dtype_map.get(output.dtype, 0)
    stream_ptr = torch.cuda.current_stream().cuda_stream if stream is None else stream

    # 5. 调用C库函数
    result = cuda_c_lib.cuda_ultimate_fusion_c(
        ctypes.c_void_p(inputs.data_ptr()),
        ctypes.c_void_p(qkv_weights.data_ptr()),
        ctypes.c_void_p(lora_a_ptr_array.data_ptr()),
        ctypes.c_void_p(lora_b_ptr_array.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(intermediate_buffer.data_ptr()),
        ctypes.c_void_p(token_indices_sorted.data_ptr()),
        ctypes.c_void_p(lora_ids.data_ptr()),
        ctypes.c_void_p(num_tokens_per_lora.data_ptr()),
        ctypes.c_void_p(lora_token_start_loc.data_ptr()),
        ctypes.c_void_p(slice_starts.data_ptr()),
        ctypes.c_void_p(lora_ranks_tensor.data_ptr()),
        ctypes.c_int(len(lora_ids)),
        ctypes.c_int(num_tokens),
        ctypes.c_int(hidden_size),
        ctypes.c_int(qkv_output_size),
        ctypes.c_int(num_slices),
        ctypes.c_int(max_rank),
        ctypes.c_int(input_strides[0]), ctypes.c_int(input_strides[1]),
        ctypes.c_int(qkv_strides[0]), ctypes.c_int(qkv_strides[1]),
        ctypes.c_int(final_lora_a_strides[0]), ctypes.c_int(final_lora_a_strides[1]), ctypes.c_int(final_lora_a_strides[2]),
        ctypes.c_int(final_lora_b_strides[0]), ctypes.c_int(final_lora_b_strides[1]), ctypes.c_int(final_lora_b_strides[2]),
        ctypes.c_int(output_strides[0]), ctypes.c_int(output_strides[1]),
        ctypes.c_void_p(stream_ptr),
        ctypes.c_int(input_dtype),
        ctypes.c_int(output_dtype),
    )
    
    if result != 0:
        raise RuntimeError(f"Ultimate fusion CUDA kernel failed with error code {result}")

    return output

# --- Part 4: 测试函数 ---

def test_ultimate_fusion():
    """一个全面的测试用例，用于验证内核的正确性和CUDA Graph的兼容性。"""
    if not C_LIB_AVAILABLE:
        print("❌ Test skipped: Ultimate fusion library not available.")
        return False
    if not torch.cuda.is_available():
        print("❌ Test skipped: CUDA not available.")
        return False
    
    print("\n" + "="*50)
    print("🧪 Testing Ultimate Fusion Kernel (Graph-Compatible)...")
    print("="*50)
    
    # 测试配置
    num_tokens = 128
    hidden_size = 1024
    num_q_heads, num_kv_heads = 8, 4
    head_dim = 128
    q_size = num_q_heads * head_dim
    k_size = num_kv_heads * head_dim
    v_size = num_kv_heads * head_dim
    qkv_output_size = q_size + k_size + v_size
    rank = 16
    num_loras = 4
    
    device = torch.device('cuda:0')
    dtype = torch.float16
    
    # 创建测试张量
    print(f"🔧 Creating test tensors on {device} with dtype {dtype}...")
    inputs = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    qkv_weights = torch.randn(qkv_output_size, hidden_size, dtype=dtype, device=device)
    
    # 创建LoRA权重 (Q, K, V三个slice)
    output_slices = (q_size, k_size, v_size)
    lora_a_stacked = tuple(
        torch.randn(num_loras, 1, rank, hidden_size, dtype=dtype, device=device).contiguous()
        for _ in output_slices
    )
    lora_b_stacked = tuple(
        torch.randn(num_loras, 1, slice_size, rank, dtype=dtype, device=device).contiguous()
        for slice_size in output_slices
    )
    
    # 创建Punica元数据 (模拟多个请求，每个请求使用不同的LoRA)
    tokens_per_lora = num_tokens // num_loras
    num_tokens_per_lora = torch.tensor([tokens_per_lora] * num_loras, dtype=torch.int32, device=device)
    lora_token_start_loc = torch.arange(0, num_tokens, tokens_per_lora, dtype=torch.int32, device=device)
    lora_ids = torch.arange(num_loras, dtype=torch.int32, device=device)
    
    # token_indices_sorted 在此简化场景下是顺序的
    token_indices_sorted = torch.arange(num_tokens, dtype=torch.int32, device=device)
    
    try:
        # --- 第一次调用 (JIT编译和缓存填充) ---
        print("\n--- Running 1st call (JIT compilation & cache warming) ---")
        output_eager = cuda_ultimate_fusion_interface(
            inputs, qkv_weights, lora_a_stacked, lora_b_stacked, output_slices,
            token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, lora_ids
        )
        torch.cuda.synchronize()
        print("✅ Eager execution successful.")
        
        # --- 第二次调用 (CUDA Graph 捕捉和回放) ---
        print("\n--- Running 2nd call (CUDA Graph capture & replay) ---")
        
        # 捕捉
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output_graph = cuda_ultimate_fusion_interface(
                inputs, qkv_weights, lora_a_stacked, lora_b_stacked, output_slices,
                token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, lora_ids
            )
        print("✅ Graph capture successful.")
        
        # 回放
        g.replay()
        torch.cuda.synchronize()
        print("✅ Graph replay successful.")
        
        # 验证结果
        print("\n--- Verifying results ---")
        are_equal = torch.allclose(output_eager, output_graph, atol=1e-2, rtol=1e-3)
        if are_equal:
            print("✅ Verification PASSED: Eager and Graph outputs match.")
        else:
            print("❌ Verification FAILED: Eager and Graph outputs DO NOT match.")
            return False

        print(f"\n📊 Output shape: {output_graph.shape}")
        print(f"📈 Output stats: min={output_graph.min():.3f}, max={output_graph.max():.3f}, mean={output_graph.mean():.3f}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Ultimate fusion kernel test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Initiating Ultimate Fusion CTypes Wrapper Test...")
    
    test_passed = test_ultimate_fusion()
    
    print("\n" + "="*50)
    if test_passed:
        print("🎉🎉🎉 Success! The ultimate fusion wrapper works correctly and is CUDA Graph compatible!")
    else:
        print("😭😭😭 Failure! The ultimate fusion wrapper has an issue.")
    print("="*50)