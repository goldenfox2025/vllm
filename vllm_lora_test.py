#!/usr/bin/env python3
"""
Simple vLLM LoRA test script with minimal configuration
"""

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def main():
    # Model paths
    base_model_path = "hf_models/Qwen2.5-1.5B"
    lora_model_path = "hf_models/Qwen2.5-1.5B-lora1"
    
    print("=== Simple vLLM LoRA Test ===")
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_model_path}")
    print("")
    
    # Initialize LLM with minimal configuration
    print("Initializing vLLM...")
    import os
    os.environ["VLLM_USE_V1"] = "0"  # Force V0 engine

    llm = LLM(
        model=base_model_path,
        enable_lora=True,
        max_lora_rank=128,  # Match LoRA model rank
        max_loras=1,
        max_model_len=256,  # Very small context
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,  # Increase memory usage
        enforce_eager=True,  # Disable compilation
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        max_num_seqs=4,  # Limit batch size
    )
    print("vLLM initialized successfully!")
    
    # Create LoRA request
    lora_request = LoRARequest("lora1", 1, lora_model_path)
    
    # Simple sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=10,    # Very short
    )
    
    # Simple test prompt
    prompts = ["Hello"]
    
    print("Running warmup...")
    # Warmup without LoRA
    warmup_outputs = llm.generate(prompts, sampling_params)
    print("Warmup completed.")
    
    print("Running LoRA inference...")
    # LoRA inference - this should trigger the triton kernels
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    
    print("Results:")
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
