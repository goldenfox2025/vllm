#!/usr/bin/env python3
"""
Script to run vLLM with LoRA and capture triton kernels using NCU
"""

import os
import sys
import time
import subprocess
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def main():
    # Model paths
    base_model_path = "hf_models/Qwen2.5-1.5B"
    lora_model_path = "hf_models/Qwen2.5-1.5B-lora1"
    
    print("Starting vLLM with LoRA...")
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_model_path}")
    
    # Initialize LLM with LoRA support - working configuration
    import os
    os.environ["VLLM_USE_V1"] = "0"  # Force V0 engine

    llm = LLM(
        model=base_model_path,
        enable_lora=True,
        max_lora_rank=64,  # Match LoRA model rank
        max_loras=1,
        max_model_len=256,  # Small context for faster execution
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,  # High memory usage
        enforce_eager=True,  # Disable compilation for faster startup
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        max_num_seqs=4,  # Limit batch size
    )
    
    # Create LoRA request
    lora_request = LoRARequest("lora1", 1, lora_model_path)
    
    # Sampling parameters - reduced for faster execution
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=10,    # Very short for faster execution
    )
    
    # Test prompts
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "What are the benefits of renewable energy?",
    ]
    
    print("\nWarming up the model...")
    # Warmup without LoRA first
    warmup_outputs = llm.generate(prompts[:2], sampling_params)
    print("Warmup completed.")
    
    print("\nStarting LoRA inference (this is where NCU should capture)...")
    
    # This is where the LoRA kernels will be called
    # NCU should be configured to start capturing around this point
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    
    print("\nGenerated outputs:")
    for i, output in enumerate(outputs):
        print(f"\nPrompt {i+1}: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
    
    print("\nLoRA inference completed.")

if __name__ == "__main__":
    main()
