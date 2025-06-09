#!/bin/bash

# NCU分析QKV+LoRA融合内核性能脚本

echo "🔍 启动NCU内核性能分析"
echo "=" * 80

# 设置环境变量
export VLLM_ENABLE_QKV_LORA_FUSION=1
export VLLM_ENABLE_LORA_TIMING=1  
export VLLM_FORCE_TRITON_LORA=1  # 强制使用Triton便于分析
export VLLM_USE_V1=0

echo "🔧 环境变量设置:"
echo "   VLLM_ENABLE_QKV_LORA_FUSION=$VLLM_ENABLE_QKV_LORA_FUSION"
echo "   VLLM_ENABLE_LORA_TIMING=$VLLM_ENABLE_LORA_TIMING"
echo "   VLLM_FORCE_TRITON_LORA=$VLLM_FORCE_TRITON_LORA"
echo

# 检查NCU是否可用
if ! command -v ncu &> /dev/null; then
    echo "❌ NCU (Nsight Compute) 未安装或不在PATH中"
    echo "💡 请安装NVIDIA Nsight Compute"
    exit 1
fi

echo "✅ NCU工具检测成功"
echo

# NCU分析选项
NCU_METRICS="sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
NCU_OUTPUT="qkv_lora_fusion_analysis"

echo "🚀 开始NCU分析..."
echo "📊 分析指标: $NCU_METRICS"
echo "📁 输出文件: ${NCU_OUTPUT}.ncu-rep"
echo

# 运行NCU分析
ncu --target-processes all \
    --metrics $NCU_METRICS \
    --export ${NCU_OUTPUT} \
    --force-overwrite \
    --print-summary per-kernel \
    python test_qkv_lora_timing.py

echo
echo "🎉 NCU分析完成!"
echo "📊 结果文件: ${NCU_OUTPUT}.ncu-rep"
echo "💡 使用以下命令查看详细报告:"
echo "   ncu-ui ${NCU_OUTPUT}.ncu-rep" 