# QKV+LoRA融合性能测量指南

## 概述

本指南介绍如何使用新实现的性能测量功能来评估QKV+LoRA融合优化的效果。通过详细的GPU计时和对比分析，您可以：

1. **测量融合优化的实际加速比**
2. **对比Triton vs CUDA kernel性能**
3. **使用NCU分析具体的kernel耗时**
4. **验证数值正确性**

## 环境变量控制

### 核心控制变量

```bash
# 启用QKV+LoRA融合优化
export VLLM_ENABLE_QKV_LORA_FUSION=1

# 启用详细性能测量
export VLLM_ENABLE_LORA_TIMING=1

# 强制使用Triton（用于性能对比）
export VLLM_FORCE_TRITON_LORA=1  # 或0
```

### 变量说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `VLLM_ENABLE_QKV_LORA_FUSION` | 0 | 启用融合优化 |
| `VLLM_ENABLE_LORA_TIMING` | 0 | 启用详细计时 |
| `VLLM_FORCE_TRITON_LORA` | 0 | 强制使用Triton kernel |

## 性能测量模式

### 1. 传统方法测量

```
传统流程: QKV matmul → LoRA Shrink → LoRA Expand
           ↓            ↓             ↓
测量项目:  QKV时间    Shrink时间    Expand时间
```

### 2. 融合方法测量

```
融合流程: 融合matmul (QKV+LoRA Shrink) → LoRA Expand
          ↓                              ↓
测量项目: 融合Matmul时间                 Expand时间
```

## 使用方法

### 方法1：使用测试脚本

```bash
# 1. 运行完整性能测试
python test_qkv_lora_performance.py

# 2. 只测试Triton版本
VLLM_FORCE_TRITON_LORA=1 python test_qkv_lora_performance.py

# 3. 测试CUDA vs Triton对比
VLLM_FORCE_TRITON_LORA=0 python test_qkv_lora_performance.py
```

### 方法2：集成到现有代码

```python
import os

# 启用性能测量
os.environ["VLLM_ENABLE_QKV_LORA_FUSION"] = "1"
os.environ["VLLM_ENABLE_LORA_TIMING"] = "1"

# 创建LLM并运行推理
llm = LLM(model="your_model", enable_lora=True)
outputs = llm.generate(prompts, lora_request=lora_request)
```

## 性能报告解读

### 示例输出

```
📈 [性能报告] QKV+LoRA计算性能对比
================================================================================
🔵 传统方法 (QKV + LoRA Shrink + LoRA Expand):
   QKV计算:      2.143 ms
   LoRA Shrink:  0.856 ms (估计)
   LoRA Expand:  1.284 ms (估计)
   总计:         4.283 ms

🟢 融合方法 (Fused QKV+LoRA + LoRA Expand):
   融合Matmul:   1.876 ms (QKV+LoRA Shrink)
   LoRA Expand:  1.234 ms
   总计:         3.110 ms

⚡ 性能提升:
   总体加速比:   1.38x
   时间节省:     1.173 ms (27.4%)

🔍 详细分析:
   计算阶段对比: 2.999ms → 1.876ms (加速 1.60x)
   Expand阶段对比: 1.284ms → 1.234ms
   ✅ 融合优化有效！减少了 37.4% 的计算时间
================================================================================
```

### 关键指标说明

1. **总体加速比**：融合方法相对传统方法的整体加速
2. **计算阶段加速比**：QKV+Shrink阶段的加速效果
3. **时间节省**：绝对时间减少量和百分比

## NCU分析

### 1. 传统方法的kernel分析

使用`VLLM_FORCE_TRITON_LORA=1`强制使用Triton，然后运行NCU：

```bash
# 设置环境
export VLLM_ENABLE_QKV_LORA_FUSION=0  # 禁用融合
export VLLM_FORCE_TRITON_LORA=1

# 运行NCU分析
ncu --set full -o traditional_lora python your_test.py
```

**期望看到的kernel**：
- `lora_shrink_*` - LoRA Shrink kernel
- `lora_expand_*` - LoRA Expand kernel  
- 常规的matmul kernel（用于QKV）

### 2. 融合方法的kernel分析

```bash
# 设置环境
export VLLM_ENABLE_QKV_LORA_FUSION=1   # 启用融合
export VLLM_ENABLE_LORA_TIMING=0       # 禁用计时以减少噪音

# 运行NCU分析
ncu --set full -o fused_lora python your_test.py
```

**期望看到的kernel**：
- 大的`torch.matmul` kernel（融合了QKV+LoRA Shrink）
- `lora_expand_*` - LoRA Expand kernel
- **没有**独立的`lora_shrink_*` kernel

### 3. 对比分析

```bash
# 对比两个报告
ncu-ui traditional_lora.ncu-rep fused_lora.ncu-rep
```

**关注指标**：
- **Kernel数量**：融合方法应该减少kernel启动
- **总GPU时间**：融合方法应该更短
- **内存带宽利用率**：融合方法可能更高

## Triton vs CUDA Kernel对比

### 强制使用Triton

```bash
export VLLM_FORCE_TRITON_LORA=1
python test_qkv_lora_performance.py
```

### 使用CUDA Kernel（带验证）

```bash
export VLLM_FORCE_TRITON_LORA=0  
python test_qkv_lora_performance.py
```

系统会自动：
1. 运行Triton版本
2. 运行CUDA版本  
3. 对比结果正确性
4. 如果差异过大会退出

## 故障排除

### 1. 融合优化未生效

**症状**：看到传统QKV调试信息
**原因**：可能是验证过程的调试输出
**解决**：设置`VLLM_ENABLE_LORA_TIMING=0`来禁用验证

### 2. 性能提升不明显

**可能原因**：
- 输入规模太小，kernel启动开销不明显
- 内存带宽瓶颈
- GPU利用率已经很高

**建议**：
- 增加batch size或sequence length
- 使用更大的LoRA rank
- 检查GPU利用率

### 3. CUDA kernel失败

**症状**：看到"CUDA LoRA kernel failed"
**解决**：
- 设置`VLLM_FORCE_TRITON_LORA=1`强制使用Triton
- 检查CUDA编译是否正确

## 最佳实践

### 1. 性能测试建议

- **Warmup**：至少运行3次热身
- **多次测量**：取10次运行的平均值
- **稳定环境**：固定GPU频率，关闭其他GPU任务

### 2. 测试场景

推荐测试不同的：
- **Batch sizes**: 1, 4, 8, 16
- **Sequence lengths**: 128, 512, 1024, 2048  
- **LoRA ranks**: 16, 32, 64, 128

### 3. 结果验证

始终检查：
- 数值正确性验证通过
- 没有NaN或Inf值
- 输出与传统方法一致

## 总结

通过本指南的方法，您可以：

1. ✅ **量化融合优化效果** - 获得准确的加速比数据
2. ✅ **识别性能瓶颈** - 了解哪个阶段最耗时
3. ✅ **对比不同实现** - Triton vs CUDA性能差异
4. ✅ **验证正确性** - 确保优化不影响结果准确性
5. ✅ **NCU深度分析** - kernel级别的性能洞察

这些工具为QKV+LoRA融合优化提供了完整的性能评估体系。 