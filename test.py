import torch
import torch.nn.functional as F
import time


dtype = torch.bfloat16  
iterations = 100
warmup_iters = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def time_operation(name: str, func, *args):
    """一个封装了预热、JIT编译、同步和计时的辅助函数"""
    print(f"--- 正在测试: {name} ---")
    
    # 预热并触发JIT编译
    print("  -> 预热并触发JIT编译...")
    for _ in range(warmup_iters):
        _ = func(*args)
    torch.cuda.synchronize(device)
    
    # 精确计时
    print("  -> 开始精确计时...")
    torch.cuda.synchronize(device)
    start_time = time.time()
    for _ in range(iterations):
        _ = func(*args)
    torch.cuda.synchronize(device)
    end_time = time.time()
    
    elapsed_ms = (end_time - start_time) * 1000
    print(f"  ✅ {name} 耗时: {elapsed_ms:.4f} ms\n")
    return elapsed_ms


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("未找到CUDA设备，测试无法在GPU上运行。")
        exit()
    if not torch.cuda.is_bf16_supported():
        print(f"当前GPU '{torch.cuda.get_device_name(0)}' 不支持 bfloat16，请使用 float16 或 float32。")
        exit()
        
    print(f"--- 性能测试将在 '{device}' 上运行 ---")
    print(f"--- JIT编译器: torch.compile 已启用, 数据类型: {dtype} ---\n")

    # 定义并编译计算函数

    def compiled_linear(x, w, b):
        return F.linear(x, w, b)


    def compiled_matmul(x, w_t):
        return torch.matmul(x, w_t)

    # 创建bfloat16张量
    input_tensor = torch.randn(20, 1536, device=device, dtype=dtype)
    weight_linear = torch.randn(2048, 1536, device=device, dtype=dtype)
    bias_linear = torch.randn(2048, device=device, dtype=dtype)
    weight_matmul = torch.randn(2048 + 384, 1536, device=device, dtype=dtype)

    # 依次调用计时函数
    time_operation(
        "场景A (JIT-compiled F.linear)",
        compiled_linear,
        input_tensor, weight_linear, bias_linear
    )
    
    time_operation(
        "场景B (JIT-compiled Matmul)",
        compiled_matmul,
        input_tensor, weight_matmul.T
    )