#!/bin/bash

# vLLM 基础模型及多LoRA并发自动化基准测试脚本
# 请根据您的实际环境调整以下变量

# --- 配置变量 ---
VLLM_ENV_NAME="base" # 您运行vLLM的Conda环境名称，如果不是base请修改
VLLM_PROJECT_DIR="/home/hzwang/vllm" # vLLM项目代码的根目录
MODEL_PATH="/home/hzwang/vllm/hf_models/Qwen2.5-7B"
TOKENIZER_PATH="/home/hzwang/vllm/hf_models/Qwen2.5-7B"
RESULTS_BASE_DIR="${VLLM_PROJECT_DIR}/benchmark_multilora_results_fp8_ext_lora_rank_32_no_eos" # 测试结果保存的基础目录
GLOBAL_SUMMARY_FILE="${RESULTS_BASE_DIR}/all_multilora_tests_summary_with_units.txt" # 全局汇总摘要文件
SERVER_LOG_FILE_TEMPLATE="${VLLM_PROJECT_DIR}/vllm_server_stage_{STAGE_ID}.log" # 服务日志模板
SERVER_HOST="0.0.0.0"
SERVER_PORT="8000"
NUM_PROMPTS=200 # 每个测试用例的提示数量
GPU_MEMORY_UTILIZATION="0.92"
DTYPE="bfloat16"
KV_CACHE_DTYPE="fp8"
QUANTIZATION="fp8"
PID_FILE="${RESULTS_BASE_DIR}/.vllm_server.pid" # PID文件路径

# --- 快速测试模式配置 ---
QUICK_TEST_MODE=false
MAX_CONCURRENT_LORAS_IN_QUICK_TEST=1 # 在快速测试中，除了基础模型，额外测试加载1个LoRA的场景

# --- 所有可用的 LoRA 配置 ---
declare -a ALL_AVAILABLE_LORAS=(
    "qwen_lora1=${VLLM_PROJECT_DIR}/hf_models/Qwen2.5-7B-lora1"
    "qwen_lora2=${VLLM_PROJECT_DIR}/hf_models/Qwen2.5-7B-lora2"
    "qwen_lora3=${VLLM_PROJECT_DIR}/hf_models/Qwen2.5-7B-lora3"
    "qwen_lora4=${VLLM_PROJECT_DIR}/hf_models/Qwen2.5-7B-lora4"
    "qwen_lora5=${VLLM_PROJECT_DIR}/hf_models/Qwen2.5-7B-lora5"
)
MAX_TOTAL_AVAILABLE_LORAS=${#ALL_AVAILABLE_LORAS[@]}

# 测试矩阵 (bs, seqlen_total) - 可能会被快速测试模式覆盖
declare -a BATCH_SIZES=(32 64 128 256)
declare -a SEQ_LENS=(512 1024 2048 4096)

# 将在主循环中动态设置
declare -a LORA_MODULES_FOR_SERVER=() # 用于服务器 --lora-modules 参数 (name=path 形式)
MAX_LORAS_FOR_SERVER_FLAG=1          # 用于服务器 --max-loras 参数
CURRENT_SERVER_LOG_FILE=""
VLLM_SERVER_PID=""

# --- 解析命令行参数 ---
if [[ "$1" == "--quick-test" ]]; then
    echo ">>> 快速测试模式已激活 <<<"
    QUICK_TEST_MODE=true
fi

# 如果是快速测试模式，则覆盖测试参数
if [ "${QUICK_TEST_MODE}" = "true" ]; then
    echo "快速测试模式：正在覆盖测试参数..."
    BATCH_SIZES=(1)       # 最小的批处理大小
    SEQ_LENS=(128)        # 最小的合理序列长度
    NUM_PROMPTS=2         # 最少的提示数量
    echo "快速测试参数：BS=${BATCH_SIZES[*]}, SL=${SEQ_LENS[*]}, Prompts=${NUM_PROMPTS}"
fi


# --- 函数定义 ---

# 启动 vLLM API 服务器
# 参数: $1 = 当前测试阶段的标识 (用于日志文件名)
start_vllm_server() {
    local stage_id_for_log=$1
    # LORA_MODULES_FOR_SERVER 和 MAX_LORAS_FOR_SERVER_FLAG 应该是全局设置好的
    
    CURRENT_SERVER_LOG_FILE="${SERVER_LOG_FILE_TEMPLATE//\{STAGE_ID\}/$stage_id_for_log}"
    rm -f "${PID_FILE}" 

    echo ">>> 正在启动 vLLM API 服务器 (阶段: ${stage_id_for_log}, 通过--lora-modules加载 ${#LORA_MODULES_FOR_SERVER[@]} 个LoRA, --max-loras ${MAX_LORAS_FOR_SERVER_FLAG})..."
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
        --enable-lora # 始终启用LoRA能力
        --max-loras "${MAX_LORAS_FOR_SERVER_FLAG}"
    )

    if [ ${#LORA_MODULES_FOR_SERVER[@]} -gt 0 ]; then
        local lora_modules_arg_values=""
        for lora_config_str in "${LORA_MODULES_FOR_SERVER[@]}"; do
            lora_modules_arg_values+="${lora_config_str} "
        done
        server_cmd_args+=(--lora-modules ${lora_modules_arg_values% }) # 去掉末尾空格
        echo "将通过 --lora-modules 加载以下LoRA到服务器: ${lora_modules_arg_values% }"
    else
        echo "服务器将不通过 --lora-modules 加载任何特定LoRA (例如基础模型测试阶段)。"
    fi
    
    echo "准备启动服务器命令: ${server_cmd_args[*]}" >&2 
    
    (
        if [ -n "${VLLM_ENV_NAME}" ]; then
            echo "Subshell (PID $$): Attempting to activate conda env: ${VLLM_ENV_NAME}" >&2
            CONDA_BASE_PATH=$(conda info --base 2>/dev/null) 
            if [ -n "${CONDA_BASE_PATH}" ] && [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ]; then
                # shellcheck source=/dev/null
                source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
                conda activate "${VLLM_ENV_NAME}"
                if [ $? -ne 0 ]; then
                    echo "Subshell (PID $$): WARN - Failed to activate conda environment '${VLLM_ENV_NAME}'. Proceeding." >&2
                else
                    echo "Subshell (PID $$): Conda env '${VLLM_ENV_NAME}' successfully activated." >&2
                fi
            else
                 echo "Subshell (PID $$): WARN - conda.sh not found or conda base not determined. Cannot activate conda env." >&2
            fi
        fi
        export VLLM_ATTENTION_BACKEND=FLASHINFER
        echo "Subshell (PID $$): Executing server command: ${server_cmd_args[*]}" >&2
        exec "${server_cmd_args[@]}"
    ) > "${CURRENT_SERVER_LOG_FILE}" 2>&1 &

    VLLM_SERVER_PID=$! 
    sleep 0.5 

    if [ -z "${VLLM_SERVER_PID}" ] || ! [[ "${VLLM_SERVER_PID}" =~ ^[0-9]+$ ]] ; then
        echo "错误：未能捕获到有效的 vLLM API 服务器的 PID！捕获到的值: '${VLLM_SERVER_PID}'." >&2
        echo "请检查服务器日志: ${CURRENT_SERVER_LOG_FILE}" >&2; tail -n 50 "${CURRENT_SERVER_LOG_FILE}" >&2
        return 1
    fi
    if ! ps -p "${VLLM_SERVER_PID}" > /dev/null; then
        echo "错误：捕获到的PID ${VLLM_SERVER_PID} 对应的进程已不存在。服务器可能启动失败。" >&2
        echo "请检查服务器日志: ${CURRENT_SERVER_LOG_FILE}" >&2; tail -n 50 "${CURRENT_SERVER_LOG_FILE}" >&2
        return 1
    fi

    echo "${VLLM_SERVER_PID}" > "${PID_FILE}"
    echo "服务器已启动，PID ${VLLM_SERVER_PID} (写入 ${PID_FILE})。日志: ${CURRENT_SERVER_LOG_FILE}" >&2
    echo "服务器启动命令 (实际执行): ${server_cmd_args[*]}"

    echo ">>> 等待服务器启动 (最多等待180秒)..."
    for i in {1..180}; do
        if ! ps -p "${VLLM_SERVER_PID}" > /dev/null; then 
            echo "错误：服务器进程 PID ${VLLM_SERVER_PID} 在等待期间消失。请检查 ${CURRENT_SERVER_LOG_FILE}"; tail -n 50 "${CURRENT_SERVER_LOG_FILE}"
            return 1
        fi
        if curl -s --max-time 5 "http://localhost:${SERVER_PORT}/health" > /dev/null; then
            echo "服务器已就绪！"
            return 0
        fi
        sleep 1
    done
    echo "错误：服务器在180秒内未能成功启动或响应健康检查。请检查 ${CURRENT_SERVER_LOG_FILE}"; tail -n 50 "${CURRENT_SERVER_LOG_FILE}"
    return 1
} # End of start_vllm_server

# 停止 vLLM API 服务器
stop_vllm_server() {
    local pid_to_kill="${VLLM_SERVER_PID}"
    if [ -z "${pid_to_kill}" ] && [ -s "${PID_FILE}" ]; then
        echo "VLLM_SERVER_PID为空，尝试从 ${PID_FILE} 读取PID..." >&2
        pid_to_kill_from_file=$(cat "${PID_FILE}")
        if [[ "${pid_to_kill_from_file}" =~ ^[0-9]+$ ]]; then pid_to_kill="${pid_to_kill_from_file}"; else
            echo "从PID文件读取的PID无效: '${pid_to_kill_from_file}'" >&2; fi
    fi

    if [ -n "${pid_to_kill}" ] && [[ "${pid_to_kill}" =~ ^[0-9]+$ ]]; then
        echo ">>> 正在停止 vLLM API 服务器 (PID: ${pid_to_kill})..."
        if ps -p "${pid_to_kill}" > /dev/null; then 
            kill "${pid_to_kill}"
            for i in {1..10}; do
                if ! ps -p "${pid_to_kill}" > /dev/null; then echo "服务器已停止."; VLLM_SERVER_PID=""; rm -f "${PID_FILE}"; return 0; fi
                sleep 1
            done
            echo "服务器未能正常停止 (kill)，尝试强制停止 (kill -9)..."; kill -9 "${pid_to_kill}"; sleep 1 
            if ! ps -p "${pid_to_kill}" > /dev/null; then echo "服务器已强制停止."; else echo "警告：服务器未能被强制停止."; fi
        else echo "服务器进程 PID ${pid_to_kill} 已不存在."; fi
    else
        if [ -z "${pid_to_kill}" ]; then
            echo "没有活动的服务器PID记录，无需停止。(在脚本正常结束时，这是预期行为)" >&2
        else
            echo "服务器PID无效 ('${pid_to_kill}')，无法执行停止操作。" >&2
        fi
    fi
    VLLM_SERVER_PID=""; rm -f "${PID_FILE}" 
} # End of stop_vllm_server

# 运行单个基准测试
# 参数: $1=bs, $2=total_seqlen, $3=current_results_dir
# 参数: $4=client_lora_modules_arg (一个包含0个或多个LoRA名称的字符串，用空格分隔，用于客户端的 --lora-modules)
run_benchmark_case() {
    local bs=$1
    local total_seqlen=$2
    local current_results_output_dir=$3 
    local client_lora_modules_arg="$4" # 注意：参数用引号包围以处理空格

    local input_len=$((total_seqlen / 2))
    local output_len=$((total_seqlen - input_len))

    local test_type_log
    local result_filename_suffix

    if [ -z "${client_lora_modules_arg}" ]; then
        test_type_log="Base Model"
        result_filename_suffix="_base"
    else
        # 计算传递给客户端的LoRA数量
        local num_client_loras=$(echo "${client_lora_modules_arg}" | wc -w)
        if [ "${num_client_loras}" -eq 1 ]; then
            test_type_log="Client requests LoRA: ${client_lora_modules_arg}"
            result_filename_suffix="_lora_${client_lora_modules_arg}"
        else
            test_type_log="Client requests LoRA Mix (${num_client_loras} adapters): ${client_lora_modules_arg}"
            # 为了文件名简洁，使用 _client_lora_mix，具体LoRA列表在日志中
            result_filename_suffix="_client_lora_mix" 
        fi
    fi
    
    echo ""
    echo ">>> 正在运行测试 (${test_type_log}): Batch Size = ${bs}, Total Seqlen = ${total_seqlen} (Input = ${input_len}, Output = ${output_len})"

    local result_filename_prefix="bs${bs}_seq${total_seqlen}${result_filename_suffix}"
    local current_timestamp; current_timestamp=$(date +%Y%m%d-%H%M%S)
    local result_json_filename="${result_filename_prefix}_${current_timestamp}.json"
    local full_json_path="${current_results_output_dir}/${result_json_filename}"

    mkdir -p "${current_results_output_dir}"

    echo "--------------------------------------------------" >> "${GLOBAL_SUMMARY_FILE}"
    echo "Test Case (${test_type_log}): Batch Size = ${bs}, Total Seqlen = ${total_seqlen} (Input = ${input_len}, Output = ${output_len})" >> "${GLOBAL_SUMMARY_FILE}"
    echo "Server PID at test time: ${VLLM_SERVER_PID:-"Unknown"}" >> "${GLOBAL_SUMMARY_FILE}" 
    echo "Timestamp: ${current_timestamp}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "Individual JSON: ${current_results_output_dir#${VLLM_PROJECT_DIR}/}/${result_json_filename}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "--------------------------------------------------" >> "${GLOBAL_SUMMARY_FILE}"

    local client_cmd_args=(
        python benchmarks/benchmark_serving.py
        --host localhost --port "${SERVER_PORT}" --model "${MODEL_PATH}" --tokenizer "${TOKENIZER_PATH}"
        --dataset-name random --random-input-len "${input_len}" --random-output-len "${output_len}"
        --num-prompts "${NUM_PROMPTS}" --max-concurrency "${bs}" --trust-remote-code
        --save-result --result-dir "${current_results_output_dir}" --result-filename "${result_json_filename}"
        --percentile-metrics "ttft,tpot,itl,e2el,decode_iteration_time,pure_computation_time"
        --ignore-eos
    )
    # 如果 client_lora_modules_arg 不为空，则将其作为 --lora-modules 的参数
    # benchmark_serving.py 的 --lora-modules nargs="+" 可以接受空格分隔的多个值
    if [ -n "${client_lora_modules_arg}" ]; then 
        client_cmd_args+=(--lora-modules ${client_lora_modules_arg}) # 不再需要额外的引号
    fi
    
    (
        if [ -n "${VLLM_ENV_NAME}" ]; then
            CONDA_BASE_PATH=$(conda info --base 2>/dev/null)
            if [ -n "${CONDA_BASE_PATH}" ] && [ -f "${CONDA_BASE_PATH}/etc/profile.d/conda.sh" ]; then
                # shellcheck source=/dev/null
                source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"; conda activate "${VLLM_ENV_NAME}"; fi
        fi
        cd "${VLLM_PROJECT_DIR}" && "${client_cmd_args[@]}" >> "${GLOBAL_SUMMARY_FILE}" 
    )
    local benchmark_exit_code=$?
    echo "" >> "${GLOBAL_SUMMARY_FILE}" 

    if [ ${benchmark_exit_code} -ne 0 ]; then
        echo "错误：测试 (${test_type_log}) bs=${bs}, seqlen=${total_seqlen} 失败."; echo "错误：测试 (${test_type_log}) bs=${bs}, seqlen=${total_seqlen} 失败。" >> "${GLOBAL_SUMMARY_FILE}"
    else
        echo "测试完成 (${test_type_log}): bs=${bs}, seqlen=${total_seqlen}."; echo "原始JSON保存在: ${full_json_path}"
    fi
} # End of run_benchmark_case

# --- 主逻辑 ---
mkdir -p "${RESULTS_BASE_DIR}" 

echo ">>> vLLM 多LoRA并发基准测试汇总摘要 - $(date)" > "${GLOBAL_SUMMARY_FILE}"
if [ "${QUICK_TEST_MODE}" = "true" ]; then echo ">>> 执行模式: 快速测试 <<<" >> "${GLOBAL_SUMMARY_FILE}"; fi
echo "测试配置: Model=${MODEL_PATH}, GPU_Mem_Util=${GPU_MEMORY_UTILIZATION}, DType=${DTYPE}, KV_Cache=${KV_CACHE_DTYPE}, Quant=${QUANTIZATION}" >> "${GLOBAL_SUMMARY_FILE}"
echo "所有可用LoRA适配器 (用于多LoRA阶段): ${ALL_AVAILABLE_LORAS[*]}" >> "${GLOBAL_SUMMARY_FILE}"
echo "PID 文件将保存在: ${PID_FILE}" >> "${GLOBAL_SUMMARY_FILE}"
echo "==================================================" >> "${GLOBAL_SUMMARY_FILE}"; echo "" >> "${GLOBAL_SUMMARY_FILE}"

trap stop_vllm_server EXIT SIGINT SIGTERM

# 确定测试阶段的迭代次数 (0 for base, 1 for 1 LoRA, etc.)
effective_max_lora_loading_stages="${MAX_TOTAL_AVAILABLE_LORAS}"
if [ "${QUICK_TEST_MODE}" = "true" ]; then
    effective_max_lora_loading_stages=${MAX_CONCURRENT_LORAS_IN_QUICK_TEST} 
fi

# 主测试循环: 0代表基础模型, >0 代表服务器上通过 --lora-modules 加载的LoRA数量
for num_loras_to_load_on_server in $(seq 0 "${effective_max_lora_loading_stages}"); do
    stage_description=""
    server_log_stage_identifier="" 

    if [ "${num_loras_to_load_on_server}" -eq 0 ]; then
        # ---- 基础模型测试阶段 ----
        stage_description="基础模型"
        server_log_stage_identifier="base_model"
        LORA_MODULES_FOR_SERVER=() 
        MAX_LORAS_FOR_SERVER_FLAG="${MAX_TOTAL_AVAILABLE_LORAS}" 
        if [ "${MAX_LORAS_FOR_SERVER_FLAG}" -eq 0 ]; then MAX_LORAS_FOR_SERVER_FLAG=1; fi
        CURRENT_ROUND_RESULTS_DIR="${RESULTS_BASE_DIR}/base_model_test"
    else
        # ---- 多LoRA测试阶段 ----
        stage_description="${num_loras_to_load_on_server} 个LoRA并发 (服务器加载)"
        server_log_stage_identifier="${num_loras_to_load_on_server}_loras_loaded"
        LORA_MODULES_FOR_SERVER=("${ALL_AVAILABLE_LORAS[@]:0:${num_loras_to_load_on_server}}")
        MAX_LORAS_FOR_SERVER_FLAG="${num_loras_to_load_on_server}"
        CURRENT_ROUND_RESULTS_DIR="${RESULTS_BASE_DIR}/concurrent_${num_loras_to_load_on_server}_loras"
    fi

    echo ""; echo "######################################################################"
    echo ">>> 开始测试阶段: ${stage_description} <<<"
    echo "######################################################################"; echo ""
    mkdir -p "${CURRENT_ROUND_RESULTS_DIR}"

    echo "==================================================" >> "${GLOBAL_SUMMARY_FILE}"
    echo "测试阶段: ${stage_description}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "服务器LoRA配置: --max-loras ${MAX_LORAS_FOR_SERVER_FLAG}" >> "${GLOBAL_SUMMARY_FILE}"
    if [ ${#LORA_MODULES_FOR_SERVER[@]} -gt 0 ]; then
        echo "服务器通过 --lora-modules 加载: ${LORA_MODULES_FOR_SERVER[*]}" >> "${GLOBAL_SUMMARY_FILE}"
    else
        echo "服务器不通过 --lora-modules 加载特定LoRA。" >> "${GLOBAL_SUMMARY_FILE}"
    fi
    echo "结果子目录: ${CURRENT_ROUND_RESULTS_DIR#${VLLM_PROJECT_DIR}/}" >> "${GLOBAL_SUMMARY_FILE}"
    echo "==================================================" >> "${GLOBAL_SUMMARY_FILE}"; echo "" >> "${GLOBAL_SUMMARY_FILE}"

    start_vllm_server "${server_log_stage_identifier}"
    if [ $? -ne 0 ]; then
        echo "错误：为阶段 '${stage_description}' 启动服务器失败。跳过此阶段测试。"
        stop_vllm_server 
        continue 
    fi
    
    if [ "${num_loras_to_load_on_server}" -eq 0 ]; then
        # ---- 执行基础模型基准测试 (客户端不指定LoRA) ----
        echo ""; echo ">>> 正在执行 ${stage_description} 的基准测试 (客户端请求基础模型)..."
        for bs_val in "${BATCH_SIZES[@]}"; do
            for seq_val in "${SEQ_LENS[@]}"; do
                run_benchmark_case "${bs_val}" "${seq_val}" "${CURRENT_ROUND_RESULTS_DIR}" "" # 第四个参数为空
                if [ "${QUICK_TEST_MODE}" = "false" ]; then echo ">>> 短暂休眠10秒..."; sleep 10; fi
            done
        done
    else
        # ---- 执行多LoRA混合请求基准测试 ----
        # 客户端的 --lora-modules 参数将包含所有当前在服务器上加载的LoRA的名称
        client_lora_names_for_benchmark_arg=""
        for lora_cfg_str_with_path in "${LORA_MODULES_FOR_SERVER[@]}"; do
            lora_name_only="${lora_cfg_str_with_path%%=*}" # 从 "name=path" 中提取 name
            client_lora_names_for_benchmark_arg+="${lora_name_only} "
        done
        # 移除末尾可能存在的空格，并确保参数不为空字符串（如果只有一个LoRA，末尾可能有空格）
        client_lora_names_for_benchmark_arg=$(echo "${client_lora_names_for_benchmark_arg}" | xargs) 

        echo ""; echo ">>> 正在执行 ${stage_description} 的基准测试 (客户端请求随机混合以下LoRA: ${client_lora_names_for_benchmark_arg})..."
        echo "--------------------------------------------------" >> "${GLOBAL_SUMMARY_FILE}"
        echo "客户端请求LoRA混合: ${client_lora_names_for_benchmark_arg} (服务器加载LoRA数: ${num_loras_to_load_on_server})" >> "${GLOBAL_SUMMARY_FILE}"
        echo "--------------------------------------------------" >> "${GLOBAL_SUMMARY_FILE}"; echo "" >> "${GLOBAL_SUMMARY_FILE}"

        for bs_val in "${BATCH_SIZES[@]}"; do
            for seq_val in "${SEQ_LENS[@]}"; do
                run_benchmark_case "${bs_val}" "${seq_val}" "${CURRENT_ROUND_RESULTS_DIR}" "${client_lora_names_for_benchmark_arg}"
                 if [ "${QUICK_TEST_MODE}" = "false" ]; then echo ">>> 短暂休眠10秒..."; sleep 10; fi
            done
        done
    fi
    echo ">>> ${stage_description} 的所有基准测试已完成！"

    stop_vllm_server
    
    if [ "${QUICK_TEST_MODE}" = "false" ]; then
      echo ">>> 完成 ${stage_description} 测试阶段。休眠30秒后继续..."
      sleep 30
    else
      echo ">>> 完成 ${stage_description} 快速测试阶段。"
    fi
done # End of main test loop

echo ""; echo "######################################################################"
echo ">>> 所有基准测试已完成！"
echo "######################################################################"
echo "所有原始JSON结果保存在基础目录: ${RESULTS_BASE_DIR}"
echo "所有测试用例的汇总摘要保存在: ${GLOBAL_SUMMARY_FILE}"
if [ "${QUICK_TEST_MODE}" = "true" ]; then echo "注意: 本次为快速测试模式运行。"; fi

exit 0
