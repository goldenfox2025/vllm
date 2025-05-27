#!/usr/bin/env python3
"""
分析continuous timing实验结果 - 改进版本
1. compute和overhead分开画图
2. 展示更多迭代数据（不限制在前200个）
3. 验证overhead计算方式
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

def parse_filename(filename):
    """从文件名中提取batch_size和num_loras"""
    pattern = r'continuous_timing_bs(\d+)_loras(\d+)_'
    match = re.search(pattern, filename)
    if match:
        batch_size = int(match.group(1))
        num_loras = int(match.group(2))
        return batch_size, num_loras
    return None, None

def load_experiment_data(file_path):
    """加载单个实验文件的时间序列数据"""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        if record.get('type') != 'experiment_start' and 'iteration' in record:
                            data.append(record)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data

def validate_data_consistency(experiments):
    """验证数据一致性：检查 step_total_time ≈ compute_time + overhead_time"""
    print("\n验证数据一致性 (step_total_time ≈ compute_time + overhead_time):")
    print("="*70)
    
    for key, exp in list(experiments.items())[:3]:  # 检查前3个实验
        step_times = exp['step_times'][:10]  # 前10个数据点
        compute_times = exp['compute_times'][:10]
        overhead_times = exp['overhead_times'][:10]
        
        print(f"\n{key} (前10个迭代):")
        for i in range(min(5, len(step_times))):  # 只显示前5个
            calculated = compute_times[i] + overhead_times[i]
            diff = abs(step_times[i] - calculated)
            print(f"  迭代{i+1}: step={step_times[i]:.2f}, compute={compute_times[i]:.2f}, "
                  f"overhead={overhead_times[i]:.2f}, 计算和={calculated:.2f}, 差值={diff:.2f}")

def collect_time_series_data():
    """收集所有符合条件的时间序列数据"""
    data_dir = Path('../continuous_timing_logs')
    target_batch_sizes = [32, 64, 128, 256]
    target_loras = [0, 1, 2, 3, 4, 5]
    
    all_experiments = {}
    
    for file_path in data_dir.glob('continuous_timing_bs*_loras*_*.jsonl'):
        batch_size, num_loras = parse_filename(file_path.name)
        
        if batch_size in target_batch_sizes and num_loras in target_loras:
            print(f"Processing: {file_path.name} (bs={batch_size}, loras={num_loras})")
            
            experiment_data = load_experiment_data(file_path)
            
            if experiment_data and len(experiment_data) > 10:  # 确保有足够的数据点
                # 提取时间序列
                iterations = [d['iteration'] for d in experiment_data]
                step_times = [d['step_total_time_ms'] for d in experiment_data]
                compute_times = [d['compute_time_ms'] for d in experiment_data]
                overhead_times = [d['overhead_time_ms'] for d in experiment_data]
                
                key = f"BS{batch_size}_LoRA{num_loras}"
                all_experiments[key] = {
                    'batch_size': batch_size,
                    'num_loras': num_loras,
                    'iterations': iterations,
                    'step_times': step_times,
                    'compute_times': compute_times,
                    'overhead_times': overhead_times,
                    'file_name': file_path.name
                }
    
    return all_experiments

def create_comprehensive_time_series_plots(experiments):
    """创建全面的时间序列图表"""
    
    batch_sizes = [32, 64, 128, 256]
    
    # 1. 步骤总时间折线图 - 展示更多数据点
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle('Step Total Time Trends (Full Data)', fontsize=16)
    
    for idx, bs in enumerate(batch_sizes):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        colors = plt.cm.Set3(np.linspace(0, 1, 6))
        
        for lora_count in range(6):
            key = f"BS{bs}_LoRA{lora_count}"
            if key in experiments:
                exp = experiments[key]
                # 使用更多数据点 - 每100个点取1个来避免过密
                total_points = len(exp['iterations'])
                step_size = max(1, total_points // 1000)  # 最多显示1000个点
                
                x = exp['iterations'][::step_size]
                y = exp['step_times'][::step_size]
                
                ax.plot(x, y, label=f'{lora_count} LoRAs', 
                       color=colors[lora_count], alpha=0.7, linewidth=1.0)
        
        ax.set_title(f'Batch Size {bs} (每{step_size}个点采样)')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Step Total Time (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('step_time_full_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 分离的计算时间图表
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle('Compute Time Trends (Full Data)', fontsize=16)
    
    for idx, bs in enumerate(batch_sizes):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        colors = plt.cm.Blues(np.linspace(0.3, 1.0, 6))
        
        for lora_count in range(6):
            key = f"BS{bs}_LoRA{lora_count}"
            if key in experiments:
                exp = experiments[key]
                total_points = len(exp['iterations'])
                step_size = max(1, total_points // 1000)
                
                x = exp['iterations'][::step_size]
                y = exp['compute_times'][::step_size]
                
                ax.plot(x, y, label=f'{lora_count} LoRAs', 
                       color=colors[lora_count], alpha=0.8, linewidth=1.2)
        
        ax.set_title(f'Batch Size {bs} - Compute Time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Compute Time (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('compute_time_full_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 分离的开销时间图表
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle('Overhead Time Trends (Full Data)', fontsize=16)
    
    for idx, bs in enumerate(batch_sizes):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        colors = plt.cm.Reds(np.linspace(0.3, 1.0, 6))
        
        for lora_count in range(6):
            key = f"BS{bs}_LoRA{lora_count}"
            if key in experiments:
                exp = experiments[key]
                total_points = len(exp['iterations'])
                step_size = max(1, total_points // 1000)
                
                x = exp['iterations'][::step_size]
                y = exp['overhead_times'][::step_size]
                
                ax.plot(x, y, label=f'{lora_count} LoRAs', 
                       color=colors[lora_count], alpha=0.8, linewidth=1.2)
        
        ax.set_title(f'Batch Size {bs} - Overhead Time')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Overhead Time (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overhead_time_full_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 吞吐量趋势（全数据）
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle('Throughput Trends (Full Data)', fontsize=16)
    
    for idx, bs in enumerate(batch_sizes):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        colors = plt.cm.Greens(np.linspace(0.3, 1.0, 6))
        
        for lora_count in range(6):
            key = f"BS{bs}_LoRA{lora_count}"
            if key in experiments:
                exp = experiments[key]
                total_points = len(exp['iterations'])
                step_size = max(1, total_points // 1000)
                
                x = exp['iterations'][::step_size]
                # 计算吞吐量
                throughput = [bs * 1000 / t for t in exp['step_times'][::step_size]]
                
                ax.plot(x, throughput, label=f'{lora_count} LoRAs', 
                       color=colors[lora_count], alpha=0.8, linewidth=1.2)
        
        ax.set_title(f'Batch Size {bs} - Throughput')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Throughput (tokens/sec)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('throughput_full_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_overhead_ratio_analysis(experiments):
    """分析overhead占比的变化趋势"""
    
    batch_sizes = [32, 64, 128, 256]
    
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle('Overhead Ratio Trends (overhead_time / step_total_time)', fontsize=16)
    
    for idx, bs in enumerate(batch_sizes):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        
        for lora_count in range(6):
            key = f"BS{bs}_LoRA{lora_count}"
            if key in experiments:
                exp = experiments[key]
                total_points = len(exp['iterations'])
                step_size = max(1, total_points // 1000)
                
                x = exp['iterations'][::step_size]
                # 计算overhead占比
                overhead_ratio = [oh/st for oh, st in zip(
                    exp['overhead_times'][::step_size], 
                    exp['step_times'][::step_size]
                )]
                
                ax.plot(x, overhead_ratio, label=f'{lora_count} LoRAs', 
                       color=colors[lora_count], alpha=0.8, linewidth=1.2)
        
        ax.set_title(f'Batch Size {bs} - Overhead Ratio')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Overhead Ratio (overhead/total)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)  # overhead占比应该在0-1之间
    
    plt.tight_layout()
    plt.savefig('overhead_ratio_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_comparison(experiments):
    """创建详细的性能对比分析"""
    
    # 计算统计数据
    summary_data = []
    for key, exp in experiments.items():
        # 使用稳定期数据（跳过前10%的warmup）
        start_idx = len(exp['step_times']) // 10
        step_times = exp['step_times'][start_idx:]
        compute_times = exp['compute_times'][start_idx:]
        overhead_times = exp['overhead_times'][start_idx:]
        
        if step_times:
            overhead_ratios = [oh/st for oh, st in zip(overhead_times, step_times)]
            
            summary_data.append({
                'batch_size': exp['batch_size'],
                'num_loras': exp['num_loras'],
                'total_iterations': len(exp['step_times']),
                'analyzed_iterations': len(step_times),
                'avg_step_time': np.mean(step_times),
                'std_step_time': np.std(step_times),
                'avg_compute_time': np.mean(compute_times),
                'std_compute_time': np.std(compute_times),
                'avg_overhead_time': np.mean(overhead_times),
                'std_overhead_time': np.std(overhead_times),
                'avg_overhead_ratio': np.mean(overhead_ratios),
                'avg_throughput': exp['batch_size'] * 1000 / np.mean(step_times),
                'cv_step_time': np.std(step_times) / np.mean(step_times)
            })
    
    df = pd.DataFrame(summary_data)
    
    # 创建对比图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. 平均时间对比
    ax = axes[0, 0]
    for bs in [32, 64, 128, 256]:
        subset = df[df['batch_size'] == bs]
        ax.plot(subset['num_loras'], subset['avg_step_time'], 
               marker='o', label=f'Total (BS {bs})', linewidth=2)
        ax.plot(subset['num_loras'], subset['avg_compute_time'], 
               marker='s', label=f'Compute (BS {bs})', linewidth=2, linestyle='--')
        ax.plot(subset['num_loras'], subset['avg_overhead_time'], 
               marker='^', label=f'Overhead (BS {bs})', linewidth=2, linestyle=':')
    ax.set_xlabel('Number of LoRAs')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Average Time Breakdown')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 2. Overhead占比
    ax = axes[0, 1]
    for bs in [32, 64, 128, 256]:
        subset = df[df['batch_size'] == bs]
        ax.plot(subset['num_loras'], subset['avg_overhead_ratio'] * 100, 
               marker='d', label=f'BS {bs}', linewidth=2)
    ax.set_xlabel('Number of LoRAs')
    ax.set_ylabel('Overhead Ratio (%)')
    ax.set_title('Average Overhead Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 吞吐量对比
    ax = axes[0, 2]
    for bs in [32, 64, 128, 256]:
        subset = df[df['batch_size'] == bs]
        ax.plot(subset['num_loras'], subset['avg_throughput'], 
               marker='h', label=f'BS {bs}', linewidth=2)
    ax.set_xlabel('Number of LoRAs')
    ax.set_ylabel('Throughput (tokens/sec)')
    ax.set_title('Average Throughput')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 稳定性对比
    ax = axes[1, 0]
    for bs in [32, 64, 128, 256]:
        subset = df[df['batch_size'] == bs]
        ax.plot(subset['num_loras'], subset['cv_step_time'], 
               marker='*', label=f'BS {bs}', linewidth=2)
    ax.set_xlabel('Number of LoRAs')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Step Time Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 数据量展示
    ax = axes[1, 1]
    for bs in [32, 64, 128, 256]:
        subset = df[df['batch_size'] == bs]
        ax.plot(subset['num_loras'], subset['total_iterations'], 
               marker='p', label=f'BS {bs}', linewidth=2)
    ax.set_xlabel('Number of LoRAs')
    ax.set_ylabel('Total Iterations')
    ax.set_title('Data Volume per Experiment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # 使用对数刻度
    
    # 6. Compute vs Overhead时间对比
    ax = axes[1, 2]
    x_pos = np.arange(len(df))
    width = 0.35
    
    ax.bar(x_pos - width/2, df['avg_compute_time'], width, 
           label='Compute Time', alpha=0.7, color='blue')
    ax.bar(x_pos + width/2, df['avg_overhead_time'], width, 
           label='Overhead Time', alpha=0.7, color='red')
    
    # 设置x轴标签
    labels = [f"BS{row['batch_size']}\nLoRA{row['num_loras']}" for _, row in df.iterrows()]
    ax.set_xticks(x_pos[::4])  # 每4个显示一个标签避免拥挤
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), 4)], rotation=45)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Compute vs Overhead Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def print_enhanced_analysis(experiments, df):
    """打印增强的分析结果"""
    print("\n" + "="*80)
    print("增强版实验分析报告")
    print("="*80)
    
    total_iterations = sum(len(exp['iterations']) for exp in experiments.values())
    print(f"\n数据量统计:")
    print(f"- 总实验配置: {len(experiments)}")
    print(f"- 总迭代次数: {total_iterations:,}")
    print(f"- 平均每配置迭代: {total_iterations // len(experiments):,}")
    
    print(f"\n各配置数据量:")
    for key, exp in experiments.items():
        print(f"- {key}: {len(exp['iterations']):,} 次迭代")
    
    print("\n" + "="*70)
    print("性能指标详细分析")
    print("="*70)
    
    print(f"{'BS':<4} {'LoRAs':<6} {'总迭代':<8} {'平均总时间':<10} {'平均计算':<10} {'平均开销':<10} {'开销占比%':<8} {'吞吐量':<8}")
    print("-" * 70)
    
    for _, row in df.sort_values(['batch_size', 'num_loras']).iterrows():
        print(f"{row['batch_size']:<4} {row['num_loras']:<6} "
              f"{row['total_iterations']:<8} {row['avg_step_time']:<10.1f} "
              f"{row['avg_compute_time']:<10.1f} {row['avg_overhead_time']:<10.1f} "
              f"{row['avg_overhead_ratio']*100:<8.1f} {row['avg_throughput']:<8.0f}")
    
    # 保存详细数据
    df.to_csv('enhanced_timing_analysis.csv', index=False)

def main():
    """主函数"""
    print("开始增强版continuous timing分析...")
    
    # 收集数据
    experiments = collect_time_series_data()
    
    if not experiments:
        print("未找到符合条件的实验数据!")
        return
    
    print(f"\n找到 {len(experiments)} 个实验配置")
    
    # 验证数据一致性
    validate_data_consistency(experiments)
    
    # 创建全面的时间序列图表
    print("\n生成全数据时间序列图表...")
    create_comprehensive_time_series_plots(experiments)
    
    # 创建overhead占比分析
    print("\n生成overhead占比分析...")
    create_overhead_ratio_analysis(experiments)
    
    # 创建详细对比分析
    print("\n生成详细性能对比...")
    df = create_detailed_comparison(experiments)
    
    # 打印增强分析
    print_enhanced_analysis(experiments, df)
    
    print("\n分析完成! 生成的文件:")
    print("- step_time_full_trends.png: 步骤总时间完整趋势")
    print("- compute_time_full_trends.png: 计算时间完整趋势")
    print("- overhead_time_full_trends.png: 开销时间完整趋势")
    print("- throughput_full_trends.png: 吞吐量完整趋势")
    print("- overhead_ratio_trends.png: overhead占比趋势")
    print("- detailed_performance_comparison.png: 详细性能对比")
    print("- enhanced_timing_analysis.csv: 增强分析数据")

if __name__ == "__main__":
    main() 