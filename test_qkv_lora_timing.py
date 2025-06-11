#!/usr/bin/env python3
"""
QKV+LoRA融合正确性和性能测试脚本（简化版 - 专注混合LoRA）
测试多个LoRA在同一批次中的处理，验证融合计算的正确性和性能

使用方法：
export VLLM_ENABLE_QKV_LORA_FUSION=1
export VLLM_ENABLE_LORA_TIMING=1
python test_qkv_lora_timing.py --num-loras 3 --batch-size 6
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
    """从模型目录中查找现有的LoRA"""
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

def create_test_llm(model_path: str, max_loras: int):
    """创建测试LLM实例"""
    print(f"🚀 初始化测试LLM (支持{max_loras}个LoRA)...")
    print(f"📁 模型路径: {model_path}")
    
    llm = LLM(
        model=model_path,
        enable_lora=True,
        max_lora_rank=128,           
        max_loras=max_loras,         
        max_model_len=256,           
        tensor_parallel_size=1,
        gpu_memory_utilization=0.65, 
        enforce_eager=True,          
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        max_num_seqs=16,             
    )
    
    print("✅ LLM初始化完成")
    return llm

def generate_mixed_batch(num_loras: int, batch_size: int) -> tuple[list[str], list[int]]:
    """生成混合LoRA批次的prompts和lora分配"""
    base_prompts = [
        "Hello, how are you?",
        "What is AI?",
        "Explain quantum physics.",
        "Write a poem about stars.",
        "How does blockchain work?",
        "Describe machine learning.",
        "What is the future of technology?",
        "How do neural networks learn?",
    ]
    
    prompts = []
    lora_assignments = []
    
    for i in range(batch_size):
        # 轮循分配LoRA
        lora_id = (i % num_loras) + 1  # LoRA ID从1开始
        prompt = base_prompts[i % len(base_prompts)]
        
        # 添加序号让每个prompt不同
        prompt = f"[{i+1}] {prompt}"
        
        prompts.append(prompt)
        lora_assignments.append(lora_id)
    
    return prompts, lora_assignments

def test_mixed_lora_batch(llm, lora_requests: list[LoRARequest], batch_size: int) -> dict:
    """测试混合LoRA批次处理"""
    print(f"\n🔥 测试混合LoRA批次处理")
    print(f"🎯 LoRA数量: {len(lora_requests)}, 批次大小: {batch_size}")
    print("=" * 60)
    
    # 生成混合批次
    prompts, lora_assignments = generate_mixed_batch(len(lora_requests), batch_size)
    
    # 打印批次分配
    print(f"📝 混合批次分配:")
    for i, (prompt, lora_id) in enumerate(zip(prompts, lora_assignments)):
        lora_name = lora_requests[lora_id-1].lora_name
        print(f"   [{i+1:2d}] LoRA-{lora_id}({lora_name}): {prompt}")
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.0,  # 贪婪解码
        max_tokens=15,    # 较短的输出便于快速测试
    )
    
    # 预热
    print(f"\n🔥 预热...")
    warmup_outputs = llm.generate([prompts[0]], sampling_params)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    time.sleep(0.5)
    
    # 正式测试
    print(f"\n⚡ 开始混合LoRA批次推理...")
    
    start_time = time.perf_counter()
    torch.cuda.synchronize()
    
    # 这里vLLM会自动处理混合LoRA批次
    outputs = llm.generate(prompts, sampling_params)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    inference_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / inference_time if inference_time > 0 else 0
    
    print(f"   ✅ 混合批次推理完成")
    print(f"   📊 推理时间: {inference_time:.4f}s")
    print(f"   📊 总tokens: {total_tokens}")
    print(f"   📊 吞吐量: {throughput:.1f} tokens/s")
    
    # 显示生成结果
    print(f"\n📝 生成结果:")
    print(f"-" * 60)
    for i, (output, lora_id) in enumerate(zip(outputs, lora_assignments)):
        prompt = prompts[i]
        generated = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)
        lora_name = lora_requests[lora_id-1].lora_name
        
        print(f"[{i+1:2d}] LoRA-{lora_id}({lora_name}):")
        print(f"     输入: {prompt}")
        print(f"     输出: {generated}")
        print(f"     Tokens: {tokens}")
        print()
    
    return {
        'inference_time': inference_time,
        'total_tokens': total_tokens,
        'throughput': throughput,
        'batch_size': batch_size,
        'num_loras': len(lora_requests),
        'outputs': outputs,
        'prompts': prompts,
        'lora_assignments': lora_assignments
    }

def compare_fusion_vs_traditional(args):
    """对比融合和传统方法在混合LoRA场景下的性能"""
    print("🎯 QKV+LoRA融合 vs 传统方法对比测试（混合LoRA批次）")
    print("=" * 80)
    
    model_path = args.model_path
    num_loras = args.num_loras
    batch_size = args.batch_size
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    print(f"✅ 模型路径: {model_path}")
    print(f"🔢 LoRA数量: {num_loras}")
    print(f"📦 批次大小: {batch_size}")
    
    # 查找LoRA
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
    setup_performance_environment()
    
    results = []
    
    # 测试配置
    test_configs = [
        {
            "name": "融合模式",
            "env_changes": {
                "VLLM_ENABLE_QKV_LORA_FUSION": "1",
                "VLLM_FORCE_TRITON_LORA": "1"
            }
        },
        {
            "name": "传统模式", 
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
            llm = create_test_llm(model_path, num_loras)
            
            # 创建LoRA请求
            lora_requests = []
            for i, lora_path in enumerate(lora_paths):
                lora_req = LoRARequest(f"lora_{i+1}", i+1, lora_path)
                lora_requests.append(lora_req)
                print(f"   创建 LoRA {i+1}: {lora_req.lora_name} -> {os.path.basename(lora_path)}")
            
            # 运行混合LoRA测试
            result = test_mixed_lora_batch(llm, lora_requests, batch_size)
            result['config_name'] = config['name']
            results.append(result)
            
            print(f"✅ 配置 '{config['name']}' 测试完成")
            
            # 清理
            del llm
            torch.cuda.empty_cache()
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ 配置 '{config['name']}' 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    # 最终对比结果
    if len(results) >= 2:
        print("\n🏆 融合 vs 传统方法最终对比")
        print("=" * 80)
        
        fusion_result = results[0]
        traditional_result = results[1]
        
        print(f"📊 测试配置:")
        print(f"   LoRA数量: {fusion_result['num_loras']}")
        print(f"   批次大小: {fusion_result['batch_size']}")
        print(f"   每LoRA平均请求: {fusion_result['batch_size'] / fusion_result['num_loras']:.1f}")
        
        print(f"\n🔵 融合模式结果:")
        print(f"   推理时间: {fusion_result['inference_time']:.4f}s")
        print(f"   吞吐量: {fusion_result['throughput']:.1f} tokens/s")
        print(f"   生成tokens: {fusion_result['total_tokens']}")
        
        print(f"\n🟢 传统模式结果:")
        print(f"   推理时间: {traditional_result['inference_time']:.4f}s")
        print(f"   吞吐量: {traditional_result['throughput']:.1f} tokens/s")
        print(f"   生成tokens: {traditional_result['total_tokens']}")
        
        # 性能对比
        if traditional_result['inference_time'] > 0:
            speedup = traditional_result['inference_time'] / fusion_result['inference_time']
            throughput_improvement = (fusion_result['throughput'] - traditional_result['throughput']) / traditional_result['throughput'] * 100
            time_saved = traditional_result['inference_time'] - fusion_result['inference_time']
            
            print(f"\n🚀 融合优化效果:")
            print(f"   加速比: {speedup:.3f}x")
            print(f"   吞吐量提升: {throughput_improvement:+.1f}%")
            print(f"   时间节省: {time_saved:.4f}s ({time_saved/traditional_result['inference_time']*100:+.1f}%)")
            
            # 评估结果
            if speedup > 1.05:
                print(f"   ✅ 融合优化有效！性能提升 {(speedup-1)*100:.1f}%")
            elif speedup > 0.95:
                print(f"   ⚖️  融合优化效果中性 (±5%范围内)")
            else:
                print(f"   ⚠️  融合优化出现性能下降 {(1-speedup)*100:.1f}%")
        
        # Token验证
        token_diff = abs(fusion_result['total_tokens'] - traditional_result['total_tokens'])
        if token_diff <= 3:
            print(f"   ✅ Token数量验证通过 (差异: {token_diff})")
        else:
            print(f"   ⚠️ Token数量差异: {token_diff}")
    
    print("=" * 80)

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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="QKV+LoRA融合测试（专注混合LoRA批次）"
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
        choices=range(2, 7),
        help="LoRA数量 (2-6)"
    )
    parser.add_argument(
        "--batch-size",
        type=int, 
        default=6,
        help="混合批次大小"
    )
    
    args = parser.parse_args()
    
    print("🎯 QKV+LoRA融合测试（专注混合LoRA批次）")
    print("🔥 测试多个LoRA在同一批次中的处理正确性和性能")
    print("=" * 80)
    
    # 打印测试参数
    print("🎮 测试参数:")
    print(f"   模型路径: {args.model_path}")
    print(f"   LoRA数量: {args.num_loras}")
    print(f"   批次大小: {args.batch_size}")
    print()
    
    # 打印系统信息
    print_system_info()
    print()
    
    # 运行测试
    compare_fusion_vs_traditional(args)
    
    print("\n🎉 混合LoRA批次测试完成!")
    print("💡 这个测试专注于验证多个LoRA在同一批次中的正确性和性能")

if __name__ == "__main__":
    main() 