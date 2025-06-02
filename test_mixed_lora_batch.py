#!/usr/bin/env python3
"""
测试vLLM的混合LoRA批处理功能，观察2048 tokens的来源
"""

import os
import sys
import torch

# 添加vLLM路径
sys.path.insert(0, '/home/vllm')

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def test_mixed_lora_batch():
    """测试混合LoRA批处理"""
    
    print("🧪 测试vLLM混合LoRA批处理")
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
        max_loras=2,       # 支持2个LoRA
        max_model_len=256,  # 适中的context
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        max_num_seqs=8,    # 增大batch size
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
        max_tokens=20,
    )
    
    # 测试1: 单个LoRA批处理
    print(f"\\n📊 测试1: 单个LoRA批处理")
    single_lora_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing.",
    ]
    
    print("🔄 处理单个LoRA批处理...")
    single_outputs = llm.generate(single_lora_prompts, sampling_params, lora_request=lora1_request)
    print(f"✅ 单个LoRA批处理完成，生成了{len(single_outputs)}个输出")
    
    # 测试2: 混合LoRA批处理（关键测试）
    print(f"📊 测试2: 混合LoRA批处理 - 这里应该看到2048 tokens!")
    mixed_prompts = [
        "Hello, how are you?",                          # LoRA 1
        "Write a short story about a robot.",           # LoRA 2
        "What is the capital of France?",               # LoRA 1
        "What are the benefits of renewable energy?",   # LoRA 2
        "Explain quantum computing in simple terms.",   # LoRA 1
        "Describe the process of photosynthesis.",      # LoRA 2
        "How does machine learning work?",              # LoRA 1
        "What is the theory of relativity?",           # LoRA 2
        "What is the theory of relativity?",  
        "What is the theory of relativity?",  
        "What is the theory of relativity?",  
        "What is the theory of relativity?",  
    ]
    
    mixed_lora_requests = [
        lora1_request,  # LoRA 1
        lora2_request,  # LoRA 2
        lora1_request,  # LoRA 1
        lora2_request,  # LoRA 2
        lora1_request,  # LoRA 1
        lora2_request,  # LoRA 2
        lora1_request,  # LoRA 1
        lora2_request,  # LoRA 2
        lora2_request,
        lora2_request,
        lora2_request,
        lora2_request,
    ]
    
    print(f"🔥 混合批处理配置:")
    for i, (prompt, lora_req) in enumerate(zip(mixed_prompts, mixed_lora_requests)):
        print(f"   [{i+1}] {lora_req.lora_name}: {prompt[:40]}...")
    
    print(f"🚀 执行混合LoRA批处理...")
    print("⚡ 这里应该触发我们的CUDA kernels并显示实际的token数量!")
    
    try:
        # 关键测试：使用list of LoRA requests
        mixed_outputs = llm.generate(
            mixed_prompts,
            sampling_params,
            lora_request=mixed_lora_requests  # 🔥 混合LoRA请求列表
        )
        
        print(f"✅ 混合LoRA批处理成功!")
        print(f"📊 生成了{len(mixed_outputs)}个输出")
        
        # 显示结果
        print(f"📋 混合批处理结果:")
        for i, (output, lora_req) in enumerate(zip(mixed_outputs, mixed_lora_requests)):
            print(f"[{lora_req.lora_name}] {output.prompt[:30]}... → {output.outputs[0].text}")
            
    except Exception as e:
        print(f"❌ 混合LoRA批处理失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 回退到顺序处理
        print("\\n🔄 回退到顺序处理...")
        mixed_outputs = []
        for prompt, lora_req in zip(mixed_prompts, mixed_lora_requests):
            output = llm.generate([prompt], sampling_params, lora_request=lora_req)
            mixed_outputs.extend(output)
    
    print(f"🎯 测试完成!")

def main():
    print("🔍 vLLM混合LoRA批处理调试工具")
    print("=" * 60)
    
    # 添加调试信息
    
    # 运行测试
    test_mixed_lora_batch()

if __name__ == "__main__":
    main()
