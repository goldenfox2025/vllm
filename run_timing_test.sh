#!/bin/bash

# QKV+LoRA融合性能时间测试启动脚本

echo "🎯 QKV+LoRA融合性能时间测试启动器"
echo "=" * 80

# 设置环境变量
export VLLM_ENABLE_QKV_LORA_FUSION=1
export VLLM_ENABLE_LORA_TIMING=1  
export VLLM_FORCE_TRITON_LORA=1
export VLLM_USE_V1=0

echo "🔧 环境变量设置:"
echo "   VLLM_ENABLE_QKV_LORA_FUSION=$VLLM_ENABLE_QKV_LORA_FUSION"
echo "   VLLM_ENABLE_LORA_TIMING=$VLLM_ENABLE_LORA_TIMING"
echo "   VLLM_FORCE_TRITON_LORA=$VLLM_FORCE_TRITON_LORA"
echo "   VLLM_USE_V1=$VLLM_USE_V1"
echo

# 检查是否存在测试LoRA
LORA_PATH="/home/vllm/hf_models/Qwen2.5-1.5B-lora1"
if [ ! -d "$LORA_PATH" ]; then
    echo "⚠️  测试LoRA不存在！"
    exit 1
fi

echo "🚀 启动性能时间测试..."
python test_qkv_lora_timing.py

echo
echo "🎉 测试完成!"
echo "💡 如果想使用NCU分析内核性能，可以运行:"
echo "   ncu --target-processes all python test_qkv_lora_timing.py" 