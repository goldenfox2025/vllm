#!/bin/bash

# ===================================================================
# Punica Kernel 全面性能测试脚本
# 整合 vLLM 原生基准测试、NCU 分析和端到端服务器测试
# ===================================================================

set -e

# === 默认配置 ===
VLLM_PROJECT_DIR="/home/hzwang/vllm"
BASE_MODEL_PATH="/home/hzwang/vllm/hf_models/Qwen2.5-7B"
LORA_MODULES=(
    "/home/hzwang/vllm/hf_models/Qwen2.5-7B-lora1"
    "/home/hzwang/vllm/hf_models/Qwen2.5-7B-lora2"
    "/home/hzwang/vllm/hf_models/Qwen2.5-7B-lora3"
    "/home/hzwang/vllm/hf_models/Qwen2.5-7B-lora4"
    "/home/hzwang/vllm/hf_models/Qwen2.5-7B-lora5"
)
CONDA_ENV="base"
PORT=8000
RESULTS_DIR="${VLLM_PROJECT_DIR}/punica_benchmark_results"

# === 测试配置 ===
# 内核级别测试参数
KERNEL_BATCH_SIZES=(1 16 32 64 128 256 512 1024)
KERNEL_HIDDEN_SIZES=(4096 8192 16384)
KERNEL_LORA_RANKS=(16 32 64)
KERNEL_NUM_LORAS=(1 2 4)
KERNEL_SEQ_LENGTHS=(1 128 512 1024)

# 端到端测试参数  
E2E_NUM_REQUESTS=100
E2E_MAX_CONCURRENCY=16
E2E_INPUT_LENGTH=256
E2E_OUTPUT_LENGTH=512

# === 性能分析配置 ===
ENABLE_NCU=false
ENABLE_KERNEL_BENCH=true
ENABLE_E2E_BENCH=true
ENABLE_CUDA_GRAPH=false
PROFILE_OUTPUT_PREFIX="punica_profile"

# === 全局变量 ===
VLLM_SERVER_PID=""
NCU_PID=""
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# === 工具函数 ===
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1" >&2
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    if [ ! -d "$VLLM_PROJECT_DIR" ]; then
        log_error "vLLM 项目目录不存在: $VLLM_PROJECT_DIR"
        exit 1
    fi
    
    if [ ! -d "$BASE_MODEL_PATH" ]; then
        log_error "基础模型路径不存在: $BASE_MODEL_PATH"
        exit 1
    fi
    
    if ! command -v python &> /dev/null; then
        log_error "未找到 python"
        exit 1
    fi
    
    # 检查 vLLM 安装
    if ! python -c "import vllm" &> /dev/null; then
        log_error "未安装 vLLM"
        exit 1
    fi
    
    # 检查 Triton
    if ! python -c "import triton" &> /dev/null; then
        log_error "未安装 Triton"
        exit 1
    fi
    
    # 检查 NCU（如果启用）
    if [ "$ENABLE_NCU" = true ] && ! command -v ncu &> /dev/null; then
        log_error "未找到 NCU，请安装 NVIDIA Nsight Compute 或禁用 NCU"
        exit 1
    fi
    
    log_info "依赖检查完成"
}

# 激活 conda 环境
activate_conda_env() {
    if [ -n "$CONDA_ENV" ]; then
        log_info "激活 conda 环境: $CONDA_ENV"
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            conda activate "$CONDA_ENV"
        else
            log_warn "无法找到 conda，跳过环境激活"
        fi
    fi
}

# 创建结果目录
setup_results_dir() {
    mkdir -p "$RESULTS_DIR"
    KERNEL_RESULTS_DIR="$RESULTS_DIR/kernel_benchmarks_$TIMESTAMP"
    E2E_RESULTS_DIR="$RESULTS_DIR/e2e_benchmarks_$TIMESTAMP"
    NCU_RESULTS_DIR="$RESULTS_DIR/ncu_profiles_$TIMESTAMP"
    
    mkdir -p "$KERNEL_RESULTS_DIR"
    mkdir -p "$E2E_RESULTS_DIR"
    mkdir -p "$NCU_RESULTS_DIR"
    
    log_info "结果将保存到: $RESULTS_DIR"
}

# 启动 vLLM 服务器
start_vllm_server() {
    local max_loras=${1:-4}
    local server_log="$E2E_RESULTS_DIR/vllm_server.log"
    
    log_info "启动 vLLM 服务器 (max_loras=$max_loras)..."
    
    local cmd_args=(
        python -m vllm.entrypoints.openai.api_server
        --model "$BASE_MODEL_PATH"
        --enable-lora
        --max-loras "$max_loras"
        --port "$PORT"
        --disable-log-requests
        --gpu-memory-utilization 0.9
        --trust-remote-code
        --swap-space 16
    )
    
    # 添加 LoRA 模块
    local lora_count=0
    for lora_path in "${LORA_MODULES[@]}"; do
        if [ $lora_count -ge $max_loras ]; then
            break
        fi
        if [ -d "$lora_path" ]; then
            cmd_args+=(--lora-modules "lora_$lora_count=$lora_path")
            ((lora_count++))
        else
            log_warn "LoRA 路径不存在，跳过: $lora_path"
        fi
    done
    
    log_info "服务器命令: ${cmd_args[*]}"
    
    (
        activate_conda_env
        export VLLM_ATTENTION_BACKEND=FLASHINFER
        cd "$VLLM_PROJECT_DIR"
        exec "${cmd_args[@]}"
    ) > "$server_log" 2>&1 &
    
    VLLM_SERVER_PID=$!
    log_info "vLLM 服务器启动，PID: $VLLM_SERVER_PID"
    
    # 等待服务器启动
    log_info "等待服务器启动..."
    local timeout=120
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if ! ps -p "$VLLM_SERVER_PID" > /dev/null; then
            log_error "服务器进程已退出"
            tail -n 20 "$server_log"
            return 1
        fi
        
        if curl -s --max-time 5 "http://localhost:$PORT/health" > /dev/null; then
            log_info "vLLM 服务器就绪"
            return 0
        fi
        
        sleep 2
        elapsed=$((elapsed + 2))
    done
    
    log_error "服务器启动超时"
    tail -n 20 "$server_log"
    return 1
}

# 停止 vLLM 服务器
stop_vllm_server() {
    if [ -n "$VLLM_SERVER_PID" ] && ps -p "$VLLM_SERVER_PID" > /dev/null; then
        log_info "停止 vLLM 服务器 (PID: $VLLM_SERVER_PID)..."
        kill "$VLLM_SERVER_PID"
        
        # 等待优雅关闭
        for i in {1..10}; do
            if ! ps -p "$VLLM_SERVER_PID" > /dev/null; then
                log_info "服务器已停止"
                VLLM_SERVER_PID=""
                return 0
            fi
            sleep 1
        done
        
        # 强制终止
        log_warn "强制终止服务器"
        kill -9 "$VLLM_SERVER_PID" 2>/dev/null || true
        VLLM_SERVER_PID=""
    fi
}

# 启动 NCU 分析
start_ncu_profiling() {
    if [ "$ENABLE_NCU" = true ] && [ -n "$VLLM_SERVER_PID" ]; then
        local ncu_output="$NCU_RESULTS_DIR/${PROFILE_OUTPUT_PREFIX}_$TIMESTAMP"
        
        log_info "启动 NCU 性能分析..."
        
        ncu \
            --target-processes all \
            --kernel-regex "(lora_expand|lora_shrink|triton.*lora.*)" \
            --metrics "sm__cycles_elapsed.avg,dram__bytes.sum,sm__sass_thread_inst_executed.sum,sm__inst_executed.sum" \
            --csv \
            --log-file "${ncu_output}.log" \
            --export "${ncu_output}.ncu-rep" \
            --profile-child-processes \
            --attach "$VLLM_SERVER_PID" &
        
        NCU_PID=$!
        log_info "NCU 分析器启动 (PID: $NCU_PID)"
        sleep 5  # 等待 NCU 启动
    fi
}

# 停止 NCU 分析
stop_ncu_profiling() {
    if [ -n "$NCU_PID" ] && ps -p "$NCU_PID" > /dev/null; then
        log_info "停止 NCU 分析器..."
        kill "$NCU_PID" 2>/dev/null || true
        wait "$NCU_PID" 2>/dev/null || true
        NCU_PID=""
        
        if [ "$ENABLE_NCU" = true ]; then
            log_info "NCU 分析结果保存在: $NCU_RESULTS_DIR"
        fi
    fi
}

# 运行内核级别基准测试
run_kernel_benchmarks() {
    if [ "$ENABLE_KERNEL_BENCH" != true ]; then
        log_info "跳过内核级别基准测试"
        return 0
    fi
    
    log_info "开始内核级别基准测试..."
    
    local benchmark_script="$VLLM_PROJECT_DIR/benchmarks/kernels/benchmark_lora.py"
    if [ ! -f "$benchmark_script" ]; then
        log_error "未找到 LoRA 基准测试脚本: $benchmark_script"
        return 1
    fi
    
    # 构建基准测试参数
    local batch_sizes_str=$(IFS=,; echo "${KERNEL_BATCH_SIZES[*]}")
    local hidden_sizes_str=$(IFS=,; echo "${KERNEL_HIDDEN_SIZES[*]}")
    local lora_ranks_str=$(IFS=,; echo "${KERNEL_LORA_RANKS[*]}")
    local num_loras_str=$(IFS=,; echo "${KERNEL_NUM_LORAS[*]}")
    local seq_lengths_str=$(IFS=,; echo "${KERNEL_SEQ_LENGTHS[*]}")
    
    local kernel_cmd_args=(
        python "$benchmark_script"
        --hidden-sizes "$hidden_sizes_str"
        --lora-ranks "$lora_ranks_str"
        --batch-sizes "$batch_sizes_str"
        --num-loras "$num_loras_str"
        --seq-lengths "$seq_lengths_str"
        --dtypes torch.bfloat16
        --sort-by-lora-ids True,False
        --expand-fn-add-inputs True,False
        --arg-pool-size 10
        --profile-directory "$KERNEL_RESULTS_DIR"
    )
    
    if [ "$ENABLE_CUDA_GRAPH" = true ]; then
        kernel_cmd_args+=(--cuda-graph-nops 10)
    fi
    
    log_info "运行内核基准测试: ${kernel_cmd_args[*]}"
    
    (
        activate_conda_env
        cd "$VLLM_PROJECT_DIR"
        exec "${kernel_cmd_args[@]}"
    ) > "$KERNEL_RESULTS_DIR/kernel_benchmark.log" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_info "内核基准测试完成"
    else
        log_error "内核基准测试失败 (退出码: $exit_code)"
        tail -n 20 "$KERNEL_RESULTS_DIR/kernel_benchmark.log"
        return 1
    fi
}

# 发送端到端测试请求
send_e2e_requests() {
    log_info "发送端到端测试请求..."
    
    # 创建临时 Python 脚本
    local e2e_script="$E2E_RESULTS_DIR/e2e_test.py"
    
    cat > "$e2e_script" << 'EOF'
#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import json
import sys
from typing import List, Optional

class E2EBenchmark:
    def __init__(self, base_url: str, lora_names: List[str]):
        self.base_url = base_url
        self.lora_names = lora_names
        self.test_prompts = [
            "请解释什么是机器学习和深度学习的区别？",
            "写一个关于人工智能的简短故事。",
            "如何优化深度神经网络的训练过程？",
            "描述卷积神经网络的工作原理。",
            "什么是注意力机制在自然语言处理中的作用？"
        ] * 20  # 扩展提示词列表
    
    async def send_single_request(self, session: aiohttp.ClientSession, 
                                 prompt: str, lora_name: Optional[str] = None):
        payload = {
            "model": lora_name if lora_name else self.lora_names[0] if self.lora_names else "base",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(sys.argv[3]),  # output_length
            "temperature": 0.7
        }
        
        start_time = time.time()
        try:
            async with session.post(f"{self.base_url}/v1/chat/completions", 
                                  json=payload, timeout=60) as response:
                result = await response.json()
                end_time = time.time()
                
                return {
                    "success": True,
                    "latency": end_time - start_time,
                    "lora": lora_name,
                    "response_length": len(result.get("choices", [{}])[0].get("message", {}).get("content", "")),
                    "prompt_length": len(prompt)
                }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "error": str(e),
                "latency": end_time - start_time,
                "lora": lora_name
            }
    
    async def run_benchmark(self, num_requests: int, max_concurrency: int):
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def controlled_request(prompt: str, lora_name: Optional[str]):
            async with semaphore:
                return await self.send_single_request(session, prompt, lora_name)
        
        tasks = []
        async with aiohttp.ClientSession() as session:
            # 分配请求到不同的 LoRA
            for i in range(num_requests):
                prompt = self.test_prompts[i % len(self.test_prompts)]
                lora_name = self.lora_names[i % len(self.lora_names)] if self.lora_names else None
                
                task = controlled_request(prompt, lora_name)
                tasks.append(task)
            
            print(f"发送 {num_requests} 个请求，最大并发: {max_concurrency}")
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # 统计结果
            successful = [r for r in results if isinstance(r, dict) and r.get("success")]
            failed = [r for r in results if isinstance(r, dict) and not r.get("success")]
            
            if successful:
                latencies = [r["latency"] for r in successful]
                avg_latency = sum(latencies) / len(latencies)
                p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
                throughput = len(successful) / (end_time - start_time)
                
                stats = {
                    "total_requests": num_requests,
                    "successful_requests": len(successful),
                    "failed_requests": len(failed),
                    "total_time": end_time - start_time,
                    "throughput_rps": throughput,
                    "avg_latency": avg_latency,
                    "p95_latency": p95_latency,
                    "min_latency": min(latencies),
                    "max_latency": max(latencies),
                    "lora_distribution": {}
                }
                
                # 按 LoRA 统计
                for lora in set([r["lora"] for r in successful]):
                    lora_results = [r for r in successful if r["lora"] == lora]
                    stats["lora_distribution"][lora or "base"] = {
                        "count": len(lora_results),
                        "avg_latency": sum(r["latency"] for r in lora_results) / len(lora_results)
                    }
                
                return stats
            else:
                return {"error": "所有请求都失败了", "failed_count": len(failed)}

async def main():
    if len(sys.argv) < 4:
        print("Usage: python e2e_test.py <num_requests> <max_concurrency> <output_length> [lora_names...]")
        sys.exit(1)
    
    num_requests = int(sys.argv[1])
    max_concurrency = int(sys.argv[2])
    lora_names = sys.argv[4:] if len(sys.argv) > 4 else []
    
    benchmark = E2EBenchmark("http://localhost:8000", lora_names)
    results = await benchmark.run_benchmark(num_requests, max_concurrency)
    
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
EOF
    
    # 构建 LoRA 名称列表
    local lora_names=()
    for i in "${!LORA_MODULES[@]}"; do
        lora_names+=("lora_$i")
    done
    
    local lora_args=""
    for name in "${lora_names[@]}"; do
        lora_args="$lora_args $name"
    done
    
    log_info "运行端到端基准测试..."
    
    (
        activate_conda_env
        cd "$E2E_RESULTS_DIR"
        python "$e2e_script" "$E2E_NUM_REQUESTS" "$E2E_MAX_CONCURRENCY" "$E2E_OUTPUT_LENGTH" $lora_args
    ) > "$E2E_RESULTS_DIR/e2e_results.json" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_info "端到端基准测试完成"
        log_info "结果保存在: $E2E_RESULTS_DIR/e2e_results.json"
    else
        log_error "端到端基准测试失败"
        cat "$E2E_RESULTS_DIR/e2e_results.json"
        return 1
    fi
}

# 运行端到端基准测试
run_e2e_benchmarks() {
    if [ "$ENABLE_E2E_BENCH" != true ]; then
        log_info "跳过端到端基准测试"
        return 0
    fi
    
    log_info "开始端到端基准测试..."
    
    # 启动服务器
    start_vllm_server ${#LORA_MODULES[@]}
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # 启动 NCU 分析（如果启用）
    start_ncu_profiling
    
    # 等待服务器稳定
    sleep 10
    
    # 发送测试请求
    send_e2e_requests
    local benchmark_result=$?
    
    # 等待 NCU 收集数据
    if [ "$ENABLE_NCU" = true ]; then
        log_info "等待 NCU 收集数据..."
        sleep 15
    fi
    
    # 停止分析和服务器
    stop_ncu_profiling
    stop_vllm_server
    
    return $benchmark_result
}

# 生成测试报告
generate_report() {
    local report_file="$RESULTS_DIR/benchmark_report_$TIMESTAMP.md"
    
    log_info "生成测试报告: $report_file"
    
    cat > "$report_file" << EOF
# Punica Kernel 基准测试报告

**测试时间**: $(date)
**测试配置**:
- 基础模型: $BASE_MODEL_PATH
- LoRA 模块数量: ${#LORA_MODULES[@]}
- 内核测试: $ENABLE_KERNEL_BENCH
- 端到端测试: $ENABLE_E2E_BENCH
- NCU 分析: $ENABLE_NCU

## 测试参数

### 内核级别测试
- Batch sizes: ${KERNEL_BATCH_SIZES[*]}
- Hidden sizes: ${KERNEL_HIDDEN_SIZES[*]}
- LoRA ranks: ${KERNEL_LORA_RANKS[*]}
- 序列长度: ${KERNEL_SEQ_LENGTHS[*]}

### 端到端测试
- 请求数量: $E2E_NUM_REQUESTS
- 最大并发: $E2E_MAX_CONCURRENCY
- 输入长度: $E2E_INPUT_LENGTH
- 输出长度: $E2E_OUTPUT_LENGTH

## 结果文件

EOF
    
    if [ "$ENABLE_KERNEL_BENCH" = true ] && [ -d "$KERNEL_RESULTS_DIR" ]; then
        echo "### 内核基准测试结果" >> "$report_file"
        echo "- 目录: $KERNEL_RESULTS_DIR" >> "$report_file"
        echo "- 日志: $KERNEL_RESULTS_DIR/kernel_benchmark.log" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    if [ "$ENABLE_E2E_BENCH" = true ] && [ -d "$E2E_RESULTS_DIR" ]; then
        echo "### 端到端基准测试结果" >> "$report_file"
        echo "- 目录: $E2E_RESULTS_DIR" >> "$report_file"
        echo "- 结果: $E2E_RESULTS_DIR/e2e_results.json" >> "$report_file"
        echo "" >> "$report_file"
        
        if [ -f "$E2E_RESULTS_DIR/e2e_results.json" ]; then
            echo "### 端到端性能摘要" >> "$report_file"
            echo '```json' >> "$report_file"
            cat "$E2E_RESULTS_DIR/e2e_results.json" >> "$report_file"
            echo '```' >> "$report_file"
            echo "" >> "$report_file"
        fi
    fi
    
    if [ "$ENABLE_NCU" = true ] && [ -d "$NCU_RESULTS_DIR" ]; then
        echo "### NCU 性能分析结果" >> "$report_file"
        echo "- 目录: $NCU_RESULTS_DIR" >> "$report_file"
        
        # 查找 NCU 结果文件
        local ncu_files=$(find "$NCU_RESULTS_DIR" -name "*.ncu-rep" 2>/dev/null)
        if [ -n "$ncu_files" ]; then
            echo "- NCU 报告文件:" >> "$report_file"
            echo "$ncu_files" | while read -r file; do
                echo "  - $file" >> "$report_file"
            done
            echo "" >> "$report_file"
            echo "使用以下命令查看 NCU 结果:" >> "$report_file"
            echo '```bash' >> "$report_file"
            echo "ncu --import $NCU_RESULTS_DIR/*.ncu-rep" >> "$report_file"
            echo '```' >> "$report_file"
        fi
    fi
    
    log_info "测试报告已生成: $report_file"
}

# 清理函数
cleanup() {
    log_info "清理资源..."
    stop_ncu_profiling
    stop_vllm_server
}

# 显示帮助信息
show_help() {
    cat << EOF
Punica Kernel 全面性能测试脚本

用法: $0 [选项]

选项:
    --base-model PATH           基础模型路径 (默认: $BASE_MODEL_PATH)
    --lora-modules PATH...      LoRA 模块路径列表
    --conda-env NAME            Conda 环境名称 (默认: $CONDA_ENV)
    --port PORT                 服务器端口 (默认: $PORT)
    --results-dir DIR           结果保存目录 (默认: $RESULTS_DIR)
    
    # 测试控制
    --enable-kernel-bench       启用内核级别基准测试 (默认: true)
    --disable-kernel-bench      禁用内核级别基准测试
    --enable-e2e-bench          启用端到端基准测试 (默认: true) 
    --disable-e2e-bench         禁用端到端基准测试
    --enable-ncu                启用 NCU 性能分析 (默认: false)
    --enable-cuda-graph         启用 CUDA Graph 测试 (默认: false)
    
    # 内核测试参数
    --kernel-batch-sizes SIZES  批次大小列表 (默认: ${KERNEL_BATCH_SIZES[*]})
    --kernel-hidden-sizes SIZES 隐藏层大小列表 (默认: ${KERNEL_HIDDEN_SIZES[*]})
    --kernel-lora-ranks RANKS    LoRA 秩列表 (默认: ${KERNEL_LORA_RANKS[*]})
    
    # 端到端测试参数
    --e2e-num-requests NUM       请求数量 (默认: $E2E_NUM_REQUESTS)
    --e2e-max-concurrency NUM    最大并发数 (默认: $E2E_MAX_CONCURRENCY)
    --e2e-output-length LEN      输出长度 (默认: $E2E_OUTPUT_LENGTH)
    
    --help                      显示此帮助信息

示例:
    # 基本测试
    $0
    
    # 只运行内核基准测试
    $0 --disable-e2e-bench
    
    # 启用 NCU 分析的完整测试
    $0 --enable-ncu
    
    # 自定义模型和 LoRA 路径
    $0 --base-model /path/to/model --lora-modules /path/to/lora1 /path/to/lora2
    
    # 快速测试（小参数）
    $0 --kernel-batch-sizes "1 16 32" --e2e-num-requests 20

EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --base-model)
                BASE_MODEL_PATH="$2"
                shift 2
                ;;
            --lora-modules)
                shift
                LORA_MODULES=()
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    LORA_MODULES+=("$1")
                    shift
                done
                ;;
            --conda-env)
                CONDA_ENV="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --results-dir)
                RESULTS_DIR="$2"
                shift 2
                ;;
            --enable-kernel-bench)
                ENABLE_KERNEL_BENCH=true
                shift
                ;;
            --disable-kernel-bench)
                ENABLE_KERNEL_BENCH=false
                shift
                ;;
            --enable-e2e-bench)
                ENABLE_E2E_BENCH=true
                shift
                ;;
            --disable-e2e-bench)
                ENABLE_E2E_BENCH=false
                shift
                ;;
            --enable-ncu)
                ENABLE_NCU=true
                shift
                ;;
            --enable-cuda-graph)
                ENABLE_CUDA_GRAPH=true
                shift
                ;;
            --kernel-batch-sizes)
                IFS=' ' read -ra KERNEL_BATCH_SIZES <<< "$2"
                shift 2
                ;;
            --kernel-hidden-sizes)
                IFS=' ' read -ra KERNEL_HIDDEN_SIZES <<< "$2"
                shift 2
                ;;
            --kernel-lora-ranks)
                IFS=' ' read -ra KERNEL_LORA_RANKS <<< "$2"
                shift 2
                ;;
            --e2e-num-requests)
                E2E_NUM_REQUESTS="$2"
                shift 2
                ;;
            --e2e-max-concurrency)
                E2E_MAX_CONCURRENCY="$2"
                shift 2
                ;;
            --e2e-output-length)
                E2E_OUTPUT_LENGTH="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 主函数
main() {
    log_info "=== Punica Kernel 全面性能测试开始 ==="
    
    # 设置信号处理
    trap cleanup EXIT INT TERM
    
    # 解析参数
    parse_args "$@"
    
    # 显示配置
    log_info "测试配置:"
    log_info "  基础模型: $BASE_MODEL_PATH"
    log_info "  LoRA 模块: ${LORA_MODULES[*]}"
    log_info "  内核测试: $ENABLE_KERNEL_BENCH"
    log_info "  端到端测试: $ENABLE_E2E_BENCH"
    log_info "  NCU 分析: $ENABLE_NCU"
    log_info "  结果目录: $RESULTS_DIR"
    
    # 检查依赖
    check_dependencies
    
    # 设置结果目录
    setup_results_dir
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 运行测试
    local test_success=true
    
    if [ "$ENABLE_KERNEL_BENCH" = true ]; then
        if ! run_kernel_benchmarks; then
            test_success=false
            log_error "内核基准测试失败"
        fi
    fi
    
    if [ "$ENABLE_E2E_BENCH" = true ]; then
        if ! run_e2e_benchmarks; then
            test_success=false
            log_error "端到端基准测试失败"
        fi
    fi
    
    # 计算总耗时
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 生成报告
    generate_report
    
    # 输出结果
    if [ "$test_success" = true ]; then
        log_info "=== 所有测试完成成功 (耗时: ${duration}s) ==="
        log_info "查看详细结果: $RESULTS_DIR"
        exit 0
    else
        log_error "=== 部分测试失败 (耗时: ${duration}s) ==="
        log_error "查看详细日志: $RESULTS_DIR"
        exit 1
    fi
}

# 运行主函数
main "$@" 