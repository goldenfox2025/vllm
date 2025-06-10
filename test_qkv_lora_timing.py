#!/usr/bin/env python3
"""
QKV+LoRA融合性能时间测试脚本（多LoRA并发版本）
支持2-6个LoRA的真正并发场景测试，模仿benchmark_serving.py的批处理方式

使用方法：
export VLLM_ENABLE_QKV_LORA_FUSION=1
export VLLM_ENABLE_LORA_TIMING=1
python test_qkv_lora_timing.py --num-loras 4 --num-requests 20 --max-tokens 50
"""

import argparse
import os
import sys
import time
import torch
import random
import shutil
import glob
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def setup_performance_environment():
    """设置性能测试环境变量"""
    print("🔧 设置性能测试环境...")
    
    # 核心性能环境变量
    performance_env = {
        "VLLM_ENABLE_QKV_LORA_FUSION": "1",  # 启用QKV+LoRA融合
        "VLLM_ENABLE_LORA_TIMING": "1",      # 启用详细时间测量
        "VLLM_USE_V1": "0",                  # 使用V0引擎
    }
    
    for key, value in performance_env.items():
        os.environ[key] = value
        print(f"   ✅ {key} = {value}")
    
    return performance_env

def find_existing_loras(model_dir: str, num_loras: int) -> list[str]:
    """从模型目录中查找现有的LoRA，不创建副本"""
    print(f"🔍 在 {model_dir} 中查找现有的 {num_loras} 个LoRA...")
    
    # 查找所有可能的LoRA目录
    possible_patterns = [
        "*lora*",
        "*LoRA*", 
        "*LORA*"
    ]
    
    found_loras = []
    for pattern in possible_patterns:
        lora_paths = glob.glob(os.path.join(model_dir, pattern))
        for path in lora_paths:
            if os.path.isdir(path):
                # 检查是否是有效的LoRA目录（包含adapter_config.json）
                if os.path.exists(os.path.join(path, "adapter_config.json")):
                    found_loras.append(path)
    
    # 去重并排序
    found_loras = sorted(list(set(found_loras)))
    
    print(f"🔍 找到 {len(found_loras)} 个LoRA:")
    for i, lora_path in enumerate(found_loras):
        print(f"   {i+1}. {os.path.basename(lora_path)}: {lora_path}")
    
    if len(found_loras) < num_loras:
        print(f"⚠️ 只找到 {len(found_loras)} 个LoRA，但需要 {num_loras} 个")
        print(f"💡 将重复使用现有LoRA以达到所需数量")
        
        # 重复使用现有LoRA直到达到所需数量
        while len(found_loras) < num_loras:
            found_loras.extend(found_loras[:min(len(found_loras), num_loras - len(found_loras))])
    
    # 只返回所需数量的LoRA
    selected_loras = found_loras[:num_loras]
    
    print(f"✅ 最终选择的 {len(selected_loras)} 个LoRA:")
    for i, lora_path in enumerate(selected_loras):
        print(f"   {i+1}. {os.path.basename(lora_path)}")
    
    return selected_loras

def create_performance_test_llm(model_path: str, max_loras: int):
    """创建优化的LLM实例用于性能测试"""
    print(f"🚀 初始化性能测试LLM (支持{max_loras}个LoRA)...")
    print(f"📁 模型路径: {model_path}")
    
    llm = LLM(
        model=model_path,
        enable_lora=True,
        max_lora_rank=128,           # 支持较大的LoRA rank
        max_loras=max_loras,         # 动态支持的LoRA数量
        max_model_len=256,           # 更小的上下文以加快测试
        tensor_parallel_size=1,
        gpu_memory_utilization=0.65, # 保守的内存使用
        enforce_eager=True,          # 禁用编译，便于测量真实性能
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        max_num_seqs=16,             # 适中的批处理大小
    )
    
    print("✅ LLM初始化完成")
    return llm

def generate_test_prompts(num_requests: int) -> list[str]:
    """生成测试prompts"""
    base_prompts = [
        "Hello, how are you today?",
        "What is the capital of France?", 
        "Explain machine learning briefly.",
        "Write a short story about a robot.",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "How do computers work?",
        "What is quantum computing?",
        "Explain artificial intelligence.",
        "How does the internet work?",
    ]
    
    # 循环使用prompts直到达到所需数量
    prompts = []
    for i in range(num_requests):
        prompt = base_prompts[i % len(base_prompts)]
        # 添加一些变化使每个请求略有不同
        if i >= len(base_prompts):
            prompt = f"Request {i+1}: {prompt}"
        prompts.append(prompt)
    
    return prompts

def measure_inference_time(llm, prompts: list[str], lora_request, sampling_params, method_name: str) -> dict:
    """精确测量推理时间的辅助函数"""
    print(f"⏱️ 测量 {method_name} 推理时间...")
    
    # 预热 - 重要！确保GPU kernels已经初始化
    print("🔥 预热GPU kernels...")
    warmup_outputs = llm.generate([prompts[0]], sampling_params, lora_request=lora_request)
    torch.cuda.synchronize()  # 确保所有操作完成
    
    # 清理
    torch.cuda.empty_cache()
    time.sleep(0.1)
    
    # 正式测量
    start_time = time.perf_counter()
    torch.cuda.synchronize()  # 开始前同步
    
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    
    torch.cuda.synchronize()  # 完成后同步
    end_time = time.perf_counter()
    
    inference_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / inference_time if inference_time > 0 else 0
    
    # 收集生成的文本
    generated_texts = []
    for i, output in enumerate(outputs):
        prompt = prompts[i] if i < len(prompts) else "N/A"
        generated_text = output.outputs[0].text
        generated_texts.append({
            'prompt': prompt,
            'generated': generated_text,
            'tokens': len(output.outputs[0].token_ids)
        })
    
    print(f"   {method_name} 时间: {inference_time:.4f}s")
    print(f"   {method_name} 总tokens: {total_tokens}")
    print(f"   {method_name} 吞吐量: {throughput:.1f} tokens/s")
    
    return {
        'time': inference_time,
        'tokens': total_tokens,
        'throughput': throughput,
        'method': method_name,
        'generated_texts': generated_texts
    }

def run_concurrent_lora_benchmark(
    llm, 
    lora_requests: list[LoRARequest], 
    num_requests: int,
    max_tokens: int,
    test_name: str
) -> dict:
    """运行真正的并发多LoRA基准测试（修正版本）"""
    print(f"\n🔥 开始并发多LoRA基准测试: {test_name}")
    print(f"🚀 LoRA数量: {len(lora_requests)}, 请求数量: {num_requests}")
    print("=" * 60)
    
    # 生成测试prompts
    test_prompts = generate_test_prompts(num_requests)
    
    # 为每个请求分配LoRA（轮询方式确保均匀分布）
    assigned_loras = []
    assigned_prompts = []
    
    for i, prompt in enumerate(test_prompts):
        lora_idx = i % len(lora_requests)
        assigned_loras.append(lora_requests[lora_idx])
        assigned_prompts.append(prompt)
        print(f"   [{i+1:2d}] {lora_requests[lora_idx].lora_name}: {prompt[:40]}...")
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.0,  # 使用贪婪解码减少随机性，便于性能对比
        max_tokens=max_tokens,
    )
    
    print(f"\n📊 测试配置:")
    print(f"   总请求数: {len(assigned_prompts)}")
    print(f"   LoRA分布: {[assigned_loras.count(lora) for lora in lora_requests]}")
    print(f"   最大生成tokens: {sampling_params.max_tokens}")
    print(f"   采样策略: 贪婪解码 (temperature=0)")
    
    # 🔥 方法1：逐个处理（测量单个推理的累积时间）
    print(f"\n⚡ 方法1: 逐个处理（基准方法）...")
    sequential_results = []
    sequential_total_time = 0
    sequential_total_tokens = 0
    
    for i, (prompt, lora_req) in enumerate(zip(assigned_prompts, assigned_loras)):
        print(f"   处理请求 {i+1}/{len(assigned_prompts)} 使用 {lora_req.lora_name}...")
        result = measure_inference_time(llm, [prompt], lora_req, sampling_params, f"Sequential-{i+1}")
        sequential_results.append(result)
        sequential_total_time += result['time']
        sequential_total_tokens += result['tokens']
    
    sequential_throughput = sequential_total_tokens / sequential_total_time if sequential_total_time > 0 else 0
    print(f"   逐个处理总时间: {sequential_total_time:.4f}s")
    print(f"   逐个处理总吞吐量: {sequential_throughput:.1f} tokens/s")
    
    # 清理GPU状态
    torch.cuda.empty_cache()
    time.sleep(1.0)  # 充分休息
    
    # 🚀 方法2：批处理多LoRA（按LoRA分组批处理）
    print(f"\n🚀 方法2: 批处理多LoRA（优化方法）...")
    
    # 按LoRA分组请求
    lora_groups = {}
    for prompt, lora_req in zip(assigned_prompts, assigned_loras):
        if lora_req.lora_name not in lora_groups:
            lora_groups[lora_req.lora_name] = {
                'lora_request': lora_req,
                'prompts': []
            }
        lora_groups[lora_req.lora_name]['prompts'].append(prompt)
    
    print(f"   分组情况:")
    for lora_name, group in lora_groups.items():
        print(f"     {lora_name}: {len(group['prompts'])} 请求")
    
    # 测量批处理时间
    batch_results = []
    batch_total_time = 0
    batch_total_tokens = 0
    
    for lora_name, group in lora_groups.items():
        print(f"   批处理 {lora_name}: {len(group['prompts'])} 请求...")
        result = measure_inference_time(
            llm, 
            group['prompts'], 
            group['lora_request'], 
            sampling_params, 
            f"Batch-{lora_name}"
        )
        batch_results.append(result)
        batch_total_time += result['time']
        batch_total_tokens += result['tokens']
    
    batch_throughput = batch_total_tokens / batch_total_time if batch_total_time > 0 else 0
    print(f"   批处理总时间: {batch_total_time:.4f}s")
    print(f"   批处理总吞吐量: {batch_throughput:.1f} tokens/s")
    
    # 性能分析
    speedup = sequential_total_time / batch_total_time if batch_total_time > 0 else 0
    throughput_improvement = (batch_throughput - sequential_throughput) / sequential_throughput * 100 if sequential_throughput > 0 else 0
    time_saved = sequential_total_time - batch_total_time
    
    print(f"\n📊 多LoRA并发性能分析:")
    print(f"   逐个处理 vs 批处理加速比: {speedup:.2f}x")
    print(f"   吞吐量提升: {throughput_improvement:+.1f}%")
    print(f"   时间节省: {time_saved:.4f}s ({time_saved/sequential_total_time*100:.1f}%)")
    
    # 验证token数量一致性
    if abs(sequential_total_tokens - batch_total_tokens) > 5:  # 允许小量差异
        print(f"⚠️ 警告：token数量不一致 (Sequential: {sequential_total_tokens}, Batch: {batch_total_tokens})")
    else:
        print(f"✅ Token数量验证通过 ({sequential_total_tokens} ≈ {batch_total_tokens})")
    
    # 📝 输出生成的句子（使用批处理结果）
    print(f"\n📝 生成的句子展示:")
    print(f"=" * 60)
    for result in batch_results:
        lora_name = result['method'].replace('Batch-', '')
        generated_texts = result.get('generated_texts', [])
        
        print(f"\n🏷️  LoRA: {lora_name}")
        print(f"-" * 40)
        
        for i, text_info in enumerate(generated_texts):
            prompt = text_info['prompt']
            generated = text_info['generated']
            tokens = text_info['tokens']
            
            # 截断过长的prompt和generated text用于显示
            prompt_display = prompt[:50] + "..." if len(prompt) > 50 else prompt
            generated_display = generated[:80] + "..." if len(generated) > 80 else generated
            
            print(f"   {i+1:2d}. 输入: {prompt_display}")
            print(f"       输出: {generated_display}")
            print(f"       Tokens: {tokens}")
            print()
    
    print(f"=" * 60)
    
    return {
        'test_name': test_name,
        'num_loras': len(lora_requests),
        'num_requests': num_requests,
        'sequential_time': sequential_total_time,
        'batch_time': batch_total_time,
        'sequential_throughput': sequential_throughput,
        'batch_throughput': batch_throughput,
        'speedup': speedup,
        'throughput_improvement': throughput_improvement,
        'time_saved': time_saved,
        'sequential_tokens': sequential_total_tokens,
        'batch_tokens': batch_total_tokens,
        'sequential_results': sequential_results,
        'batch_results': batch_results
    }

def compare_fusion_performance(args):
    """对比融合和非融合的性能，专注于多LoRA并发场景"""
    print("🎯 QKV+LoRA融合性能对比测试（专注多LoRA并发）")
    print("=" * 80)
    
    model_path = args.model_path
    num_loras = args.num_loras
    num_requests = args.num_requests
    max_tokens = args.max_tokens
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    print(f"✅ 模型路径: {model_path}")
    print(f"🔢 LoRA数量: {num_loras}")
    print(f"📝 请求数量: {num_requests}")
    print(f"🎯 最大tokens: {max_tokens}")
    
    # 查找现有的LoRA（不创建副本）
    try:
        model_dir = os.path.dirname(model_path)
        lora_paths = find_existing_loras(model_dir, num_loras)
        if not lora_paths:
            print(f"❌ 未找到足够的LoRA")
            return
    except Exception as e:
        print(f"❌ 查找LoRA失败: {e}")
        return
    
    # 设置环境
    env_config = setup_performance_environment()
    
    results = []
    
    # 简化的测试配置（专注于融合vs传统对比）
    test_configs = [
        {
            "name": f"融合模式-{num_loras}LoRA",
            "env_changes": {
                "VLLM_ENABLE_QKV_LORA_FUSION": "1",
                "VLLM_FORCE_TRITON_LORA": "1"  # 使用稳定的Triton作为基准
            }
        },
        {
            "name": f"传统模式-{num_loras}LoRA", 
            "env_changes": {
                "VLLM_ENABLE_QKV_LORA_FUSION": "0",
                "VLLM_FORCE_TRITON_LORA": "1"
            }
        }
    ]
    
    for config_idx, config in enumerate(test_configs):
        print(f"\n🔧 测试配置 {config_idx+1}/{len(test_configs)}: {config['name']}")
        print("-" * 60)
        
        # 设置环境变量
        for key, value in config['env_changes'].items():
            os.environ[key] = value
            print(f"   设置 {key} = {value}")
        
        try:
            # 创建LLM
            llm = create_performance_test_llm(model_path, num_loras)
            
            # 创建多个LoRA请求
            lora_requests = []
            for i, lora_path in enumerate(lora_paths):
                lora_req = LoRARequest(f"lora_{i+1}", i+1, lora_path)
                lora_requests.append(lora_req)
                print(f"   创建 LoRA {i+1}: {lora_req.lora_name} -> {os.path.basename(lora_path)}")
            
            # 最终预热
            print("🔥 系统级预热...")
            warmup_outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=3))
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            time.sleep(1.0)
            
            # 🔥 关键：多LoRA并发测试
            result = run_concurrent_lora_benchmark(
                llm, lora_requests, num_requests, max_tokens, config['name']
            )
            
            # 保存结果
            results.append(result)
            
            print(f"✅ 配置 '{config['name']}' 测试完成")
            
            # 彻底清理
            del llm
            torch.cuda.empty_cache()
            time.sleep(2)  # 让GPU充分休息
            
        except Exception as e:
            print(f"❌ 配置 '{config['name']}' 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    # 输出最终对比结果
    if len(results) >= 2:
        print("\n🏆 QKV+LoRA融合性能终极对比")
        print("=" * 80)
        
        fusion_result = results[0]
        traditional_result = results[1]
        
        print(f"📊 测试配置总结:")
        print(f"   LoRA数量: {fusion_result['num_loras']}")
        print(f"   请求数量: {fusion_result['num_requests']}")
        print(f"   最大tokens: {max_tokens}")
        
        print(f"\n🔵 融合模式详细结果:")
        print(f"   批处理时间: {fusion_result['batch_time']:.4f}s")
        print(f"   批处理吞吐量: {fusion_result['batch_throughput']:.1f} tokens/s")
        print(f"   内部加速比: {fusion_result['speedup']:.2f}x (顺序→批处理)")
        print(f"   生成tokens: {fusion_result['batch_tokens']}")
        
        print(f"\n🟢 传统模式详细结果:")
        print(f"   批处理时间: {traditional_result['batch_time']:.4f}s")
        print(f"   批处理吞吐量: {traditional_result['batch_throughput']:.1f} tokens/s")
        print(f"   内部加速比: {traditional_result['speedup']:.2f}x (顺序→批处理)")
        print(f"   生成tokens: {traditional_result['batch_tokens']}")
        
        # 融合 vs 传统的最终对比
        fusion_vs_traditional_speedup = traditional_result['batch_time'] / fusion_result['batch_time']
        final_throughput_improvement = (fusion_result['batch_throughput'] - traditional_result['batch_throughput']) / traditional_result['batch_throughput'] * 100
        absolute_time_saved = traditional_result['batch_time'] - fusion_result['batch_time']
        
        print(f"\n🔥 融合优化最终效果评估:")
        print(f"   融合模式批处理时间: {fusion_result['batch_time']:.4f}s")
        print(f"   传统模式批处理时间: {traditional_result['batch_time']:.4f}s")
        print(f"   🚀 融合加速比: {fusion_vs_traditional_speedup:.3f}x")
        print(f"   📈 吞吐量提升: {final_throughput_improvement:+.1f}%")
        print(f"   ⏱️  绝对时间节省: {absolute_time_saved:.4f}s")
        print(f"   📊 相对时间节省: {absolute_time_saved/traditional_result['batch_time']*100:+.1f}%")
        
        # 性能评估
        if fusion_vs_traditional_speedup > 1.05:
            print(f"   ✅ 融合优化有效！加速 {(fusion_vs_traditional_speedup-1)*100:.1f}%")
        elif fusion_vs_traditional_speedup > 0.95:
            print(f"   ⚖️  融合优化效果中性 (±5%范围内)")
        else:
            print(f"   ⚠️  融合优化出现性能下降 {(1-fusion_vs_traditional_speedup)*100:.1f}%，需要调试")
            
        # Token验证
        token_diff = abs(fusion_result['batch_tokens'] - traditional_result['batch_tokens'])
        if token_diff <= 5:
            print(f"   ✅ Token数量验证通过 (差异: {token_diff})")
        else:
            print(f"   ⚠️ Token数量差异较大: {token_diff}")

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
    parser = argparse.ArgumentParser(
        description="QKV+LoRA融合性能测试（多LoRA并发版本）"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/vllm/hf_models/Qwen2.5-1.5B",
        help="模型路径"
    )
    parser.add_argument(
        "--num-loras",
        type=int,
        default=3,
        choices=range(2, 7),  # 2-6个LoRA
        help="并发LoRA数量 (2-6)"
    )
    parser.add_argument(
        "--num-requests",
        type=int, 
        default=12,
        help="总请求数量"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="每个请求最大生成tokens"
    )
    
    args = parser.parse_args()
    
    print("🎯 QKV+LoRA融合性能测试（多LoRA并发专版）")
    print("🔥 专注于真正的多LoRA并发场景性能测试")
    print("=" * 80)
    
    # 打印测试参数
    print("🎮 测试参数:")
    print(f"   模型路径: {args.model_path}")
    print(f"   LoRA数量: {args.num_loras}")
    print(f"   请求数量: {args.num_requests}")
    print(f"   最大tokens: {args.max_tokens}")
    print()
    
    # 打印系统信息
    print_system_info()
    print()
    
    # 运行性能对比
    compare_fusion_performance(args)
    
    print("\n🎉 多LoRA并发性能测试完成!")
    print("📊 查看上面的详细性能报告以了解融合优化效果")
    print("💡 关键指标解读:")
    print("   - 🚀 融合加速比 > 1.05：有效优化")
    print("   - 📈 吞吐量提升：tokens/s的改善百分比")
    print("   - ⏱️  时间节省：绝对和相对时间改善")
    print("   - ⚖️  内部加速比：批处理相比逐个处理的效率")

if __name__ == "__main__":
    main() 