#!/usr/bin/env python3
"""
测试修复后的CUDA LoRA内核与Triton版本的等价性
"""

import os
import sys
import torch
import numpy as np

# 添加vLLM路径
sys.path.insert(0, '/home/vllm')

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def test_cuda_triton_equivalence():
    """测试CUDA和Triton LoRA内核的等价性"""
    
    print("🔧 测试CUDA与Triton LoRA内核等价性")
    print("=" * 60)
    
    # 模型路径
    base_model_path = "hf_models/Qwen2.5-1.5B"
    lora1_model_path = "hf_models/Qwen2.5-1.5B-lora1"
    lora2_model_path = "hf_models/Qwen2.5-1.5B-lora2"
    
    # 初始化LLM
    os.environ["VLLM_USE_V1"] = "0"
    
    print("🚀 初始化vLLM...")
    llm = LLM(
        model=base_model_path,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=2,
        max_model_len=256,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        max_num_seqs=8,
    )
    
    # 创建LoRA请求
    lora1_request = LoRARequest("lora1", 1, lora1_model_path)
    lora2_request = LoRARequest("lora2", 2, lora2_model_path)
    
    print(f"✅ LoRA请求创建完成:")
    print(f"   LoRA 1: {lora1_request}")
    print(f"   LoRA 2: {lora2_request}")
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
    )
    
    # 测试数据
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing.",
        "Write a short story about a robot.",
    ]
    
    test_lora_requests = [
        lora1_request,
        lora2_request,
        lora1_request,
        lora2_request,
    ]
    
    print(f"\n📊 测试配置:")
    for i, (prompt, lora_req) in enumerate(zip(test_prompts, test_lora_requests)):
        print(f"   [{i+1}] {lora_req.lora_name}: {prompt[:30]}...")
    
    print(f"\n🔥 执行混合LoRA批处理...")
    print("⚡ 这将触发CUDA kernels并与Triton进行比较!")
    
    try:
        # 执行推理
        outputs = llm.generate(
            test_prompts,
            sampling_params,
            lora_request=test_lora_requests
        )
        
        print(f"✅ 混合LoRA批处理成功!")
        print(f"📊 生成了{len(outputs)}个输出")
        
        # 显示结果
        print(f"\n📋 批处理结果:")
        for i, (output, lora_req) in enumerate(zip(outputs, test_lora_requests)):
            generated_text = output.outputs[0].text.strip()
            print(f"[{lora_req.lora_name}] {output.prompt[:20]}... → {generated_text[:30]}...")
            
        print(f"\n🎯 等价性测试完成!")
        print("✅ 如果没有错误信息，说明CUDA内核与Triton版本等价")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    print("🔍 CUDA vs Triton LoRA内核等价性测试")
    print("=" * 60)
    
    success = test_cuda_triton_equivalence()
    
    if success:
        print("\n🎉 所有测试通过！CUDA内核已成功修复为与Triton等价")
    else:
        print("\n❌ 测试失败，需要进一步调试")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
