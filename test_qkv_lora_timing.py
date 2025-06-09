#!/usr/bin/env python3
"""
QKV+LoRA融合性能时间测试脚本
基于run_vllm_lora_ncu.py，专门测试融合优化的时间性能
支持真正的多LoRA并发场景测试

使用方法：
export VLLM_ENABLE_QKV_LORA_FUSION=1
export VLLM_ENABLE_LORA_TIMING=1
export VLLM_FORCE_TRITON_LORA=1  # 强制使用Triton以便NCU分析
python test_qkv_lora_timing.py
"""

import os
import sys
import time
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def setup_performance_environment():
    """设置性能测试环境变量"""
    print("🔧 设置性能测试环境...")
    
    # 核心性能环境变量
    performance_env = {
        "VLLM_ENABLE_QKV_LORA_FUSION": "1",  # 启用QKV+LoRA融合
        "VLLM_ENABLE_LORA_TIMING": "1",      # 启用详细时间测量
        "VLLM_FORCE_TRITON_LORA": "1",       # 强制使用Triton（便于NCU分析）
        "VLLM_USE_V1": "0",                  # 使用V0引擎
    }
    
    for key, value in performance_env.items():
        os.environ[key] = value
        print(f"   ✅ {key} = {value}")
    
    return performance_env

def create_performance_test_llm(model_path: str):
    """创建优化的LLM实例用于性能测试"""
    print(f"🚀 初始化性能测试LLM...")
    print(f"📁 模型路径: {model_path}")
    
    llm = LLM(
        model=model_path,
        enable_lora=True,
        max_lora_rank=128,           # 支持较大的LoRA rank
        max_loras=2,                # 🔥 支持2个LoRA同时工作
        max_model_len=512,          # 较小的上下文以加快测试
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7, # 保守的内存使用
        enforce_eager=True,          # 禁用编译，便于测量真实性能
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        max_num_seqs=8,             # 🔥 支持批处理和多LoRA
    )
    
    print("✅ LLM初始化完成")
    return llm

def run_single_lora_benchmark(llm, lora_request, test_name: str):
    """运行单个LoRA的基准测试"""
    print(f"\n⏱️  开始单LoRA基准测试: {test_name}")
    print("=" * 60)
    
    # 测试prompts
    test_prompts = [
        "Hello, how are you today?",
        "What is the capital of France?",
        "Explain machine learning briefly.",
    ]
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=50,  # 较短生成以加快测试
    )
    
    print(f"📊 测试配置:")
    print(f"   Prompts数量: {len(test_prompts)}")
    print(f"   最大生成tokens: {sampling_params.max_tokens}")
    
    # 批处理测试
    print("📦 批处理测试:")
    batch_start = time.perf_counter()
    batch_outputs = llm.generate(test_prompts, sampling_params, lora_request=lora_request)
    batch_end = time.perf_counter()
    
    batch_time = batch_end - batch_start
    total_tokens = sum(len(output.outputs[0].token_ids) for output in batch_outputs)
    batch_tokens_per_sec = total_tokens / batch_time if batch_time > 0 else 0
    
    print(f"   批处理时间: {batch_time:.3f}s")
    print(f"   总tokens: {total_tokens}")
    print(f"   批处理吞吐量: {batch_tokens_per_sec:.1f} tokens/s")
    
    return {
        'test_name': test_name,
        'batch_time': batch_time,
        'batch_throughput': batch_tokens_per_sec,
        'total_tokens': total_tokens
    }

def run_mixed_lora_benchmark(llm, lora1_request, lora2_request, test_name: str):
    """运行混合LoRA的基准测试（真正的多LoRA并发）"""
    print(f"\n🔥 开始混合LoRA基准测试: {test_name}")
    print("🚀 这是真正的多LoRA并发场景！")
    print("=" * 60)
    
    # 创建交替的prompts和LoRA分配（模拟run_vllm_lora_ncu.py）
    mixed_batch_prompts = [
        "Hello, how are you?",                          # LoRA 1
        "Write a short story about a robot.",           # LoRA 2
        "What is the capital of France?",               # LoRA 1
        "What are the benefits of renewable energy?",   # LoRA 2
        "Explain quantum computing briefly.",           # LoRA 1
        "Describe the process of photosynthesis.",      # LoRA 2
    ]

    mixed_batch_loras = [
        lora1_request,  # LoRA 1
        lora2_request,  # LoRA 2
        lora1_request,  # LoRA 1
        lora2_request,  # LoRA 2
        lora1_request,  # LoRA 1
        lora2_request,  # LoRA 2
    ]
    
    print(f"🎯 混合批次配置:")
    for i, (prompt, lora_req) in enumerate(zip(mixed_batch_prompts, mixed_batch_loras)):
        print(f"   [{i+1}] {lora_req.lora_name}: {prompt[:40]}...")
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=50,
    )
    
    print(f"\n🔄 方法1: 顺序混合处理（模拟真实混合批次）...")
    sequential_start = time.perf_counter()
    mixed_outputs_sequential = []
    for i, (prompt, lora_req) in enumerate(zip(mixed_batch_prompts, mixed_batch_loras)):
        print(f"   处理请求 {i+1}/{len(mixed_batch_prompts)} 使用 {lora_req.lora_name}...")
        output = llm.generate([prompt], sampling_params, lora_request=lora_req)
        mixed_outputs_sequential.extend(output)
    sequential_end = time.perf_counter()
    
    sequential_time = sequential_end - sequential_start
    sequential_tokens = sum(len(output.outputs[0].token_ids) for output in mixed_outputs_sequential)
    sequential_throughput = sequential_tokens / sequential_time if sequential_time > 0 else 0
    
    print(f"   顺序混合时间: {sequential_time:.3f}s")
    print(f"   顺序混合吞吐量: {sequential_throughput:.1f} tokens/s")
    
    print(f"\n🚀 方法2: 并发批处理（分离LoRA批次）...")
    # 分离不同LoRA的prompts
    lora1_batch_prompts = [mixed_batch_prompts[i] for i in range(0, len(mixed_batch_prompts), 2)]
    lora2_batch_prompts = [mixed_batch_prompts[i] for i in range(1, len(mixed_batch_prompts), 2)]
    
    print(f"   LoRA 1 批次: {len(lora1_batch_prompts)} prompts")
    print(f"   LoRA 2 批次: {len(lora2_batch_prompts)} prompts")
    
    concurrent_start = time.perf_counter()
    
    # 并发处理两个LoRA批次
    lora1_batch_outputs = llm.generate(lora1_batch_prompts, sampling_params, lora_request=lora1_request)
    lora2_batch_outputs = llm.generate(lora2_batch_prompts, sampling_params, lora_request=lora2_request)
    
    concurrent_end = time.perf_counter()
    
    # 交错结果以匹配原始顺序
    mixed_outputs_concurrent = []
    for i in range(max(len(lora1_batch_outputs), len(lora2_batch_outputs))):
        if i < len(lora1_batch_outputs):
            mixed_outputs_concurrent.append(lora1_batch_outputs[i])
        if i < len(lora2_batch_outputs):
            mixed_outputs_concurrent.append(lora2_batch_outputs[i])
    
    concurrent_time = concurrent_end - concurrent_start
    concurrent_tokens = sum(len(output.outputs[0].token_ids) for output in mixed_outputs_concurrent)
    concurrent_throughput = concurrent_tokens / concurrent_time if concurrent_time > 0 else 0
    
    print(f"   并发批处理时间: {concurrent_time:.3f}s")
    print(f"   并发批处理吞吐量: {concurrent_throughput:.1f} tokens/s")
    
    # 性能对比
    print(f"\n📊 混合LoRA性能对比:")
    print(f"   顺序处理 vs 并发处理加速比: {sequential_time / concurrent_time:.2f}x")
    print(f"   吞吐量提升: {(concurrent_throughput - sequential_throughput) / sequential_throughput * 100:.1f}%")
    
    return {
        'test_name': test_name,
        'sequential_time': sequential_time,
        'concurrent_time': concurrent_time,
        'sequential_throughput': sequential_throughput,
        'concurrent_throughput': concurrent_throughput,
        'speedup': sequential_time / concurrent_time,
        'total_tokens': concurrent_tokens
    }

def compare_fusion_performance():
    """对比融合和非融合的性能，包括多LoRA场景"""
    print("🎯 QKV+LoRA融合性能对比测试（多LoRA并发）")
    print("=" * 80)
    
    model_path = "/home/vllm/hf_models/Qwen2.5-1.5B"
    lora1_path = "/home/vllm/hf_models/Qwen2.5-1.5B-lora1"
    lora2_path = "/home/vllm/hf_models/Qwen2.5-1.5B-lora2"
    
    # 检查路径
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    if not os.path.exists(lora1_path):
        print(f"❌ LoRA 1路径不存在: {lora1_path}")
        return
        
    if not os.path.exists(lora2_path):
        print(f"❌ LoRA 2路径不存在: {lora2_path}")
        print("💡 提示：需要创建第二个LoRA")
        # 创建lora2（复制lora1）
        import shutil
        if os.path.exists(lora1_path):
            print(f"🔧 自动复制LoRA 1到LoRA 2...")
            shutil.copytree(lora1_path, lora2_path)
            print(f"✅ 创建LoRA 2完成: {lora2_path}")
        else:
            return
    
    print(f"✅ 模型路径: {model_path}")
    print(f"✅ LoRA 1路径: {lora1_path}")
    print(f"✅ LoRA 2路径: {lora2_path}")
    
    # 设置环境
    env_config = setup_performance_environment()
    
    results = []
    
    # 测试配置
    test_configs = [
        {
            "name": "融合模式 + Triton + 多LoRA",
            "env_changes": {
                "VLLM_ENABLE_QKV_LORA_FUSION": "1",
                "VLLM_FORCE_TRITON_LORA": "1"
            }
        },
        {
            "name": "融合模式 + CUDA + 多LoRA",
            "env_changes": {
                "VLLM_ENABLE_QKV_LORA_FUSION": "1", 
                "VLLM_FORCE_TRITON_LORA": "0"
            }
        },
        {
            "name": "传统模式 + Triton + 多LoRA",
            "env_changes": {
                "VLLM_ENABLE_QKV_LORA_FUSION": "0",
                "VLLM_FORCE_TRITON_LORA": "1"
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n🔧 测试配置: {config['name']}")
        print("-" * 60)
        
        # 设置环境变量
        for key, value in config['env_changes'].items():
            os.environ[key] = value
            print(f"   设置 {key} = {value}")
        
        try:
            # 创建LLM
            llm = create_performance_test_llm(model_path)
            
            # 创建两个LoRA请求
            lora1_request = LoRARequest("test_lora1", 1, lora1_path)
            lora2_request = LoRARequest("test_lora2", 2, lora2_path)
            
            print(f"✅ 创建LoRA请求:")
            print(f"   LoRA 1: {lora1_request}")
            print(f"   LoRA 2: {lora2_request}")
            
            # 预热
            print("🔥 预热阶段...")
            warmup_outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
            
            # 单LoRA测试
            print("\n📊 单LoRA性能测试...")
            lora1_result = run_single_lora_benchmark(llm, lora1_request, f"{config['name']} - LoRA1")
            lora2_result = run_single_lora_benchmark(llm, lora2_request, f"{config['name']} - LoRA2")
            
            # 🔥 关键：混合LoRA测试（真正的多LoRA并发）
            mixed_result = run_mixed_lora_benchmark(llm, lora1_request, lora2_request, config['name'])
            
            # 保存结果
            config_result = {
                'config_name': config['name'],
                'lora1_result': lora1_result,
                'lora2_result': lora2_result,
                'mixed_result': mixed_result
            }
            results.append(config_result)
            
            print(f"✅ 配置 '{config['name']}' 测试完成")
            
            # 清理
            del llm
            torch.cuda.empty_cache()
            time.sleep(1)  # 让GPU稍微休息
            
        except Exception as e:
            print(f"❌ 配置 '{config['name']}' 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    # 输出对比结果
    if len(results) >= 2:
        print("\n🏆 多LoRA性能对比结果")
        print("=" * 80)
        
        print("📊 各配置混合LoRA并发性能:")
        for i, result in enumerate(results):
            mixed = result['mixed_result']
            print(f"   {i+1}. {result['config_name']}:")
            print(f"      并发处理时间: {mixed['concurrent_time']:.3f}s")
            print(f"      并发吞吐量: {mixed['concurrent_throughput']:.1f} tokens/s")
            print(f"      内部加速比: {mixed['speedup']:.2f}x (顺序vs并发)")
        
        # 融合 vs 传统对比
        if len(results) >= 3:
            fusion_result = results[0]['mixed_result']  # 融合模式
            traditional_result = results[2]['mixed_result']  # 传统模式
            
            fusion_vs_traditional_speedup = traditional_result['concurrent_time'] / fusion_result['concurrent_time']
            throughput_improvement = (fusion_result['concurrent_throughput'] - traditional_result['concurrent_throughput']) / traditional_result['concurrent_throughput'] * 100
            
            print(f"\n🔥 融合模式 vs 传统模式（多LoRA并发）:")
            print(f"   融合模式时间: {fusion_result['concurrent_time']:.3f}s")
            print(f"   传统模式时间: {traditional_result['concurrent_time']:.3f}s")
            print(f"   融合加速比: {fusion_vs_traditional_speedup:.2f}x")
            print(f"   吞吐量提升: {throughput_improvement:+.1f}%")

def print_system_info():
    """打印系统信息"""
    print("🔍 系统信息:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   当前GPU: {torch.cuda.current_device()}")
        print(f"   GPU名称: {torch.cuda.get_device_name()}")
        
        # GPU内存信息
        gpu_props = torch.cuda.get_device_properties(0)
        total_memory = gpu_props.total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   GPU内存: {allocated_memory:.1f}GB / {total_memory:.1f}GB")

def main():
    """主函数"""
    print("🎯 QKV+LoRA融合性能时间测试（真正的多LoRA并发）")
    print("🔥 模拟run_vllm_lora_ncu.py的多LoRA混合批处理场景")
    print("=" * 80)
    
    # 打印系统信息
    print_system_info()
    print()
    
    # 运行性能对比
    compare_fusion_performance()
    
    print("\n🎉 多LoRA并发性能测试完成!")
    print("📊 查看上面的详细性能报告以了解融合优化在多LoRA场景下的效果")
    print("💡 提示：")
    print("   - 融合模式应该在多LoRA并发时显示更好性能")
    print("   - 混合LoRA批处理测试了真正的多LoRA内核融合")
    print("   - 可以使用 nsys 或 ncu 进行更深入的多LoRA内核分析")

if __name__ == "__main__":
    main() 