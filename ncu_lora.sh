#!/bin/bash

# Simple script to run vLLM with LoRA and capture with NCU
# This will capture kernels during the entire execution but with filtering

echo "=== Simple NCU LoRA Capture ==="
echo "This will run vLLM with LoRA and capture triton kernels"
echo ""

# Create output directory
mkdir -p ncu_reports

# Generate timestamp for unique report name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_NAME="simple_lora_${TIMESTAMP}"

echo "Starting NCU capture..."
echo "Report: ncu_reports/${REPORT_NAME}.ncu-rep"
echo ""

# Run NCU with kernel filtering
# Focus on triton kernels and skip some initial launches to avoid warmup
# --kernel-name "regex:.*(lora|triton).*" \
ncu \
    --kernel-name "regex:.*(lora|triton).*" \
    --target-processes all \
    --launch-skip 500 \
    --launch-count 10 \
    --replay-mode kernel \
    --set full \
    --export "ncu_reports/${REPORT_NAME}" \
    python3 run_vllm_lora_ncu.py

echo ""
echo "=== Capture Complete ==="
echo "Report saved: ncu_reports/${REPORT_NAME}.ncu-rep"
echo ""
echo "View report:"
echo "  ncu-ui ncu_reports/${REPORT_NAME}.ncu-rep"
echo ""
echo "Generate summary:"
echo "  ncu --import ncu_reports/${REPORT_NAME}.ncu-rep --page details"
# huggingface-cli download Superrrdamn/task-13-Qwen-Qwen2.5-1.5B --local-dir .