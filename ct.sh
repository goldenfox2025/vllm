#!/bin/bash

# 简单的输出长度基准测试脚本
# 使用现有的benchmark_serving.py测试不同输出长度的LoRA性能

# --- 配置变量 ---
VLLM_ENV_NAME="base"
VLLM_PROJECT_DIR="/home/hzwang/vllm"
MODEL_PATH="/home/hzwang/vllm/hf_models/Qwen2.5-7B"
TOKENIZER_PATH="/home/hzwang/vllm/hf_models/Qwen2.5-7B"
RESULTS_BASE_DIR="${VLLM_PROJECT_DIR}/output_length_benchmark_results_continuous_timing"
GLOBAL_SUMMARY_FILE="${RESULTS_BASE_DIR}/output_length_summary.txt"
SERVER_LOG_FILE_TEMPLATE="${VLLM_PROJECT_DIR}/vllm_server_outputlen_{STAGE_ID}.log"
SERVER_HOST="0.0.0.0"
SERVER_PORT="8000"
GPU_MEMORY_UTILIZATION="0.92"
DTYPE="bfloat16"
KV_CACHE_DTYPE="fp8"
QUANTIZATION="fp8"
PID_FILE="${RESULTS_BASE_DIR}/.vllm_server.pid"
  
# --- 测试配置 ---
INPUT_LENGTH=256                    # 固定输入长度
OUTPUT_LENGTHS=(4096)  # 要测试的输出长度
BATCH_SIZES=(32 64 128 256)         # 要测试的批处理大小

# --- LoRA配置 ---
declare -a ALL_AVAILABLE_LORAS=(
    "qwen_lora1=${VLLM_PROJECT_DIR}/hf_models/Qwen2.5-7B-lora1"
    "qwen_lora2=${VLLM_PROJECT_DIR}/hf_models/Qwen2.5-7B-lora2"
    "qwen_lora3=${VLLM_PROJECT_DIR}/hf_models/Qwen2.5-7B-lora3"
    "qwen_lora4=${VLLM_PROJECT_DIR}/hf_models/Qwen2.5-7B-lora4"
    "qwen_lora5=${VLLM_PROJECT_DIR}/hf_models/Qwen2.5-7B-lora5"
)
MAX_TOTAL_AVAILABLE_LORAS=${#ALL_AVAILABLE_LORAS[@]}

# 全局变量
declare -a LORA_MODULES_FOR_SERVER=()
MAX_LORAS_FOR_SERVER_FLAG=1
CURRENT_SERVER_LOG_FILE=""
VLLM_SERVER_PID=""

# --- 快速测试模式 ---
QUICK_TEST_MODE=false
if [[ "$1" == "--quick-test" ]]; then
    echo ">>> 快速测试模式已激活 <<<"
    QUICK_TEST_MODE=true
    OUTPUT_LENGTHS=(3072 4096)
    BATCH_SIZES=(256)           # 快速测试只测试较小的批次
    NUM_PROMPTS=256                # 等于快速测试中最大的batch size
    echo "快速测试参数：输出长度=${OUTPUT_LENGTHS[*]}, 批次大小=${BATCH_SIZES[*]}, 请求数=${NUM_PROMPTS}"
fi

# --- 函数定义 ---

# 启动 vLLM API 服务器
start_vllm_server() {
    local stage_id_for_log=$1
    
    CURRENT_SERVER_LOG_FILE="${SERVER_LOG_FILE_TEMPLATE//\{STAGE_ID\}/$stage_id_for_log}"
    rm -f "${PID_FILE}" 

    echo ">>> 正在启动 vLLM API 服务器 (阶段: ${stage_id_for_log}, 加载 ${#LORA_MODULES_FOR_SERVER[@]} 个LoRA)..."
    local server_cmd_args=(
        python -m vllm.entrypoints.openai.api_server
        --model "${MODEL_PATH}"
        --tokenizer "${TOKENIZER_PATH}"
        --tensor-parallel-size 1
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        --trust-remote-code
        --dtype "${DTYPE}"
        --kv-cache-dtype "${KV_CACHE_DTYPE}"
        --quantization "${QUANTIZATION}"
        --no-enable-chunked-prefill
        --host "${SERVER_HOST}" --port "${SERVER_PORT}"
        --disable-log-requests
        --max-model-len 8192 
        --max-lora-rank 64 
        --enable-lora
        --max-loras "${MAX_LORAS_FOR_SERVER_FLAG}"
    )

    if [ ${#LORA_MODULES_FOR_SERVER[@]} -gt 0 ]; then
        local lora_modules_arg_values=""
        for lora_config_str in "${LORA_MODULES_FOR_SERVER[@]}"; do
            lora_modules_arg_values+="${lora_config_str} "
        done
        server_cmd_args+=(--lora-modules ${lora_modules_arg_values% })
        echo "将加载以下LoRA到服务器: ${lora_modules_arg_values% }"
    else
        echo "服务器将不加载任何LoRA (基础模型测试)。"
    fi
    
    (
        if [ -n "${VLLM_ENV_NAME}" ]; then
            CONDA_BASE_PATH=$(conda info --base 2>/dev/null) 
            if [ -n "${CONDA_BASE_PATH}" ] && [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ]; then
                source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
                conda activate "${VLLM_ENV_NAME}"
            fi
        fi
        export VLLM_ATTENTION_BACKEND=FLASHINFER
        exec "${server_cmd_args[@]}"
    ) > "${CURRENT_SERVER_LOG_FILE}" 2>&1 &

    VLLM_SERVER_PID=$! 
    sleep 0.5 

    if [ -z "${VLLM_SERVER_PID}" ] || ! [[ "${VLLM_SERVER_PID}" =~ ^[0-9]+$ ]] ; then
        echo "错误：未能捕获到有效的 vLLM API 服务器的 PID！"
        echo "请检查服务器日志: ${CURRENT_SERVER_LOG_FILE}"
        tail -n 20 "${CURRENT_SERVER_LOG_FILE}"
        return 1
    fi
    if ! ps -p "${VLLM_SERVER_PID}" > /dev/null; then
        echo "错误：服务器进程已不存在。服务器可能启动失败。"
        echo "请检查服务器日志: ${CURRENT_SERVER_LOG_FILE}"
        tail -n 20 "${CURRENT_SERVER_LOG_FILE}"
        return 1
    fi

    echo "${VLLM_SERVER_PID}" > "${PID_FILE}"
    echo "服务器已启动，PID ${VLLM_SERVER_PID}。日志: ${CURRENT_SERVER_LOG_FILE}"

    echo ">>> 等待服务器启动 (最多等待180秒)..."
    for i in {1..180}; do
        if ! ps -p "${VLLM_SERVER_PID}" > /dev/null; then 
            echo "错误：服务器进程在等待期间消失。"
            tail -n 20 "${CURRENT_SERVER_LOG_FILE}"
            return 1
        fi
        if curl -s --max-time 5 "http://localhost:${SERVER_PORT}/health" > /dev/null; then
            echo "服务器已就绪！"
            return 0
        fi
        sleep 1
    done
    echo "错误：服务器在180秒内未能成功启动。"
    tail -n 20 "${CURRENT_SERVER_LOG_FILE}"
    return 1
}

# 停止 vLLM API 服务器
stop_vllm_server() {
    local pid_to_kill="${VLLM_SERVER_PID}"
    if [ -z "${pid_to_kill}" ] && [ -s "${PID_FILE}" ]; then
        pid_to_kill=$(cat "${PID_FILE}")
    fi

    if [ -n "${pid_to_kill}" ] && [[ "${pid_to_kill}" =~ ^[0-9]+$ ]]; then
        echo ">>> 正在停止 vLLM API 服务器 (PID: ${pid_to_kill})..."
        if ps -p "${pid_to_kill}" > /dev/null; then 
            kill "${pid_to_kill}"
            for i in {1..10}; do
                if ! ps -p "${pid_to_kill}" > /dev/null; then 
                    echo "服务器已停止."
                    VLLM_SERVER_PID=""
                    rm -f "${PID_FILE}"
                    return 0
                fi
                sleep 1
            done
            echo "服务器未能正常停止，尝试强制停止..."
            kill -9 "${pid_to_kill}"
            sleep 1 
            if ! ps -p "${pid_to_kill}" > /dev/null; then 
                echo "服务器已强制停止."
            else 
                echo "警告：服务器未能被强制停止."
            fi
        else 
            echo "服务器进程已不存在."
        fi
    fi
    VLLM_SERVER_PID=""
    rm -f "${PID_FILE}" 
}

# 运行单个基准测试
run_benchmark_case() {
    local output_len=$1
    local batch_size=$2
    local current_results_output_dir=$3 
    local client_lora_modules_arg="$4"

    local test_type_log
    local result_filename_suffix
    local num_loras

    if [ -z "${client_lora_modules_arg}" ]; then
        test_type_log="Base Model"
        result_filename_suffix="_base"
        num_loras=0
    else
        local num_client_loras=$(echo "${client_lora_modules_arg}" | wc -w)
        test_type_log="LoRA (${num_client_loras} adapters): ${client_lora_modules_arg}"
        result_filename_suffix="_lora_${num_client_loras}"
        num_loras=${num_client_loras}
    fi
    
    echo ""
    echo ">>> 正在运行测试 (${test_type_log}): Input = ${INPUT_LENGTH}, Output = ${output_len}, Batch = ${batch_size}"

    local result_filename_prefix="input${INPUT_LENGTH}_output${output_len}_bs${batch_size}${result_filename_suffix}"
    local current_timestamp; current_timestamp=$(date +%Y%m%d-%H%M%S)
    local result_json_filename="${result_filename_prefix}_${current_timestamp}.json"
    local full_json_path="${current_results_output_dir}/${result_json_filename}"

    mkdir -p "${current_results_output_dir}"

    echo "--------------------------------------------------" >> "${GLOBAL_SUMMARY_FILE}"
    echo "Test Case (${test_type_log}): Input = ${INPUT_LENGTH}, Output = ${output_len}, Batch = ${batch_size}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "Server PID: ${VLLM_SERVER_PID:-"Unknown"}" >> "${GLOBAL_SUMMARY_FILE}" 
    echo "Timestamp: ${current_timestamp}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "JSON: ${result_json_filename}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "--------------------------------------------------" >> "${GLOBAL_SUMMARY_FILE}"

    local client_cmd_args=(
        python benchmarks/benchmark_serving.py
        --host localhost --port "${SERVER_PORT}" 
        --model "${MODEL_PATH}" --tokenizer "${TOKENIZER_PATH}"
        --dataset-name random 
        --random-input-len "${INPUT_LENGTH}" 
        --random-output-len "${output_len}"
        --num-prompts "${batch_size}" 
        --max-concurrency "${batch_size}" 
        --trust-remote-code
        --save-result 
        --result-dir "${current_results_output_dir}" 
        --result-filename "${result_json_filename}"
        --percentile-metrics "ttft,tpot,itl,e2el"
        --ignore-eos
    )
    
    # 如果有LoRA参数，添加--lora-modules
    if [ -n "${client_lora_modules_arg}" ]; then 
        client_cmd_args+=(--lora-modules ${client_lora_modules_arg})
    fi
    
    echo "执行命令: ${client_cmd_args[*]}"
    
    (
        if [ -n "${VLLM_ENV_NAME}" ]; then
            CONDA_BASE_PATH=$(conda info --base 2>/dev/null)
            if [ -n "${CONDA_BASE_PATH}" ] && [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ]; then
                source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
                conda activate "${VLLM_ENV_NAME}"
            fi
        fi
        
        cd "${VLLM_PROJECT_DIR}" && "${client_cmd_args[@]}" >> "${GLOBAL_SUMMARY_FILE}" 2>&1
    )
    local benchmark_exit_code=$?
    
    echo "" >> "${GLOBAL_SUMMARY_FILE}" 

    if [ ${benchmark_exit_code} -ne 0 ]; then
        echo "错误：测试失败 (${test_type_log}) input=${INPUT_LENGTH}, output=${output_len}, batch=${batch_size}"
        echo "错误：测试失败 (${test_type_log}) input=${INPUT_LENGTH}, output=${output_len}, batch=${batch_size}" >> "${GLOBAL_SUMMARY_FILE}"
        echo "请检查服务器日志: ${CURRENT_SERVER_LOG_FILE}"
    else
        echo "测试完成 (${test_type_log}): input=${INPUT_LENGTH}, output=${output_len}, batch=${batch_size}"
        echo "结果保存在: ${full_json_path}"
        
        # 检查是否生成了连续时间记录文件
        echo "连续时间记录文件会自动保存在 continuous_timing_logs/ 目录中"
        if [ -d "continuous_timing_logs" ]; then
            local timing_files=$(find continuous_timing_logs -name "continuous_timing_*.jsonl" -type f -newer "${full_json_path}" 2>/dev/null | head -3)
            if [ -n "${timing_files}" ]; then
                echo "最新的连续时间记录文件:"
                for file in ${timing_files}; do
                    local file_size=$(stat -c%s "${file}" 2>/dev/null || echo "0")
                    local line_count=$(wc -l < "${file}" 2>/dev/null || echo "0")
                    echo "  - ${file} (${file_size} bytes, ${line_count} lines)"
                done
            fi
        fi
    fi
}

# --- 主逻辑 ---
mkdir -p "${RESULTS_BASE_DIR}" 

echo ">>> vLLM 输出长度基准测试汇总 - $(date)" > "${GLOBAL_SUMMARY_FILE}"
if [ "${QUICK_TEST_MODE}" = "true" ]; then 
    echo ">>> 执行模式: 快速测试 <<<" >> "${GLOBAL_SUMMARY_FILE}"
fi
echo "测试配置: Model=${MODEL_PATH}" >> "${GLOBAL_SUMMARY_FILE}"
echo "输入长度: ${INPUT_LENGTH}, 输出长度: ${OUTPUT_LENGTHS[*]}, 批次大小: ${BATCH_SIZES[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "所有可用LoRA: ${ALL_AVAILABLE_LORAS[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "==================================================" >> "${GLOBAL_SUMMARY_FILE}"
echo "" >> "${GLOBAL_SUMMARY_FILE}"

trap stop_vllm_server EXIT SIGINT SIGTERM

# 主测试循环: 0代表基础模型, >0 代表加载的LoRA数量
for num_loras_to_load_on_server in $(seq 0 5); do
    stage_description=""
    server_log_stage_identifier="" 

    if [ "${num_loras_to_load_on_server}" -eq 0 ]; then
        # ---- 基础模型测试阶段 ----
        stage_description="基础模型"
        server_log_stage_identifier="base_model"
        LORA_MODULES_FOR_SERVER=() 
        MAX_LORAS_FOR_SERVER_FLAG="${MAX_TOTAL_AVAILABLE_LORAS}" 
        if [ "${MAX_LORAS_FOR_SERVER_FLAG}" -eq 0 ]; then 
            MAX_LORAS_FOR_SERVER_FLAG=1
        fi
        CURRENT_ROUND_RESULTS_DIR="${RESULTS_BASE_DIR}/base_model_test"
    else
        # ---- LoRA测试阶段 ----
        stage_description="${num_loras_to_load_on_server} 个LoRA"
        server_log_stage_identifier="${num_loras_to_load_on_server}_loras"
        LORA_MODULES_FOR_SERVER=("${ALL_AVAILABLE_LORAS[@]:0:${num_loras_to_load_on_server}}")
        MAX_LORAS_FOR_SERVER_FLAG="${num_loras_to_load_on_server}"
        CURRENT_ROUND_RESULTS_DIR="${RESULTS_BASE_DIR}/concurrent_${num_loras_to_load_on_server}_loras"
    fi

    echo ""
    echo "######################################################################"
    echo ">>> 开始测试阶段: ${stage_description} <<<"
    echo "######################################################################"
    echo ""
    mkdir -p "${CURRENT_ROUND_RESULTS_DIR}"

    echo "==================================================" >> "${GLOBAL_SUMMARY_FILE}"
    echo "测试阶段: ${stage_description}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "服务器LoRA配置: --max-loras ${MAX_LORAS_FOR_SERVER_FLAG}" >> "${GLOBAL_SUMMARY_FILE}"
    if [ ${#LORA_MODULES_FOR_SERVER[@]} -gt 0 ]; then
        echo "服务器加载LoRA: ${LORA_MODULES_FOR_SERVER[*]}" >> "${GLOBAL_SUMMARY_FILE}"
    else
        echo "服务器不加载特定LoRA。" >> "${GLOBAL_SUMMARY_FILE}"
    fi
    echo "结果目录: ${CURRENT_ROUND_RESULTS_DIR#${VLLM_PROJECT_DIR}/}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "==================================================" >> "${GLOBAL_SUMMARY_FILE}"
    echo "" >> "${GLOBAL_SUMMARY_FILE}"

    # 启动服务器
    start_vllm_server "${server_log_stage_identifier}"
    if [ $? -ne 0 ]; then
        echo "错误：为阶段 '${stage_description}' 启动服务器失败。跳过此阶段测试。"
        stop_vllm_server 
        continue 
    fi
    
    # 准备客户端LoRA参数
    client_lora_names_for_benchmark_arg=""
    if [ "${num_loras_to_load_on_server}" -gt 0 ]; then
        for lora_cfg_str_with_path in "${LORA_MODULES_FOR_SERVER[@]}"; do
            lora_name_only="${lora_cfg_str_with_path%%=*}"
            client_lora_names_for_benchmark_arg+="${lora_name_only} "
        done
        client_lora_names_for_benchmark_arg=$(echo "${client_lora_names_for_benchmark_arg}" | xargs)
    fi

    # 执行不同输出长度的测试
    echo ">>> 正在执行 ${stage_description} 的输出长度基准测试..."
    echo "客户端LoRA参数: ${client_lora_names_for_benchmark_arg:-"无(基础模型)"}"
    
    for output_len in "${OUTPUT_LENGTHS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            run_benchmark_case "${output_len}" "${batch_size}" "${CURRENT_ROUND_RESULTS_DIR}" "${client_lora_names_for_benchmark_arg}"
            
            if [ "${QUICK_TEST_MODE}" = "false" ]; then 
                echo ">>> 短暂休眠5秒..."
                sleep 5
            fi
        done
    done
    
    echo ">>> ${stage_description} 的所有输出长度测试已完成！"

    stop_vllm_server
    
    if [ "${QUICK_TEST_MODE}" = "false" ]; then
      echo ">>> 完成 ${stage_description} 测试阶段。休眠10秒后继续..."
      sleep 10
    else
      echo ">>> 完成 ${stage_description} 快速测试阶段。"
    fi
done

echo ""
echo "######################################################################"
echo ">>> 所有输出长度基准测试已完成！"
echo "######################################################################"
echo "所有结果保存在: ${RESULTS_BASE_DIR}"
echo "汇总摘要文件: ${GLOBAL_SUMMARY_FILE}"
if [ "${QUICK_TEST_MODE}" = "true" ]; then 
    echo "注意: 本次为快速测试模式运行。"
fi

exit 0 