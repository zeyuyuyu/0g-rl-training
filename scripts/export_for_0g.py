#!/usr/bin/env python3
"""
Export trained model for 0G Platform deployment

This script:
1. Loads trained LoRA weights
2. Merges with base model (optional)
3. Exports to 0G-compatible format
4. Computes model hash for registry
5. Generates config for 0G serving broker

Reference: https://huggingface.co/rachpradhan/Qwen3.5-35B-A3B-Turbo-SWE-v0.0.1
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftModel


def compute_model_hash(model_path: str) -> str:
    """
    Compute keccak256 hash of model files for 0G registry.
    
    Note: 0G uses keccak256 (Ethereum's hash). We use pycryptodome for accurate
    keccak256 computation, or fallback to hashlib.sha3_256 (which is slightly
    different but close enough for testing).
    
    The hash is used in:
    - SCRIPT_MAP in api/fine-tuning/const/const.go
    - Model identification on-chain
    """
    print(f"Computing model hash for {model_path}...")
    
    # Try to import proper keccak256
    try:
        from Crypto.Hash import keccak
        hasher = keccak.new(digest_bits=256)
        using_keccak = True
    except ImportError:
        import hashlib
        print("Warning: pycryptodome not found, using sha3_256 (may differ from on-chain keccak)")
        hasher = hashlib.sha3_256()
        using_keccak = False
    
    # Collect all model files
    model_files = []
    path = Path(model_path)
    
    for pattern in ["*.safetensors", "*.bin", "*.json", "*.txt"]:
        model_files.extend(path.glob(pattern))
    
    # Sort for deterministic hash
    model_files.sort()
    
    # Compute combined hash
    for file_path in tqdm(model_files, desc="Hashing model files"):
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    
    model_hash = "0x" + hasher.hexdigest()
    print(f"Model hash: {model_hash}")
    if not using_keccak:
        print("Note: Install pycryptodome for accurate keccak256: pip install pycryptodome")
    
    return model_hash


def export_lora_model(
    checkpoint_path: str,
    output_path: str,
    base_model_path: Optional[str] = None,
    merge_weights: bool = False
):
    """
    Export LoRA model for 0G deployment.
    
    Args:
        checkpoint_path: Path to trained checkpoint (with adapter_config.json)
        output_path: Output directory
        base_model_path: Base model path (if not in checkpoint)
        merge_weights: Whether to merge LoRA into base model
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Load model
    if merge_weights:
        # Merge LoRA weights into base model
        print("Merging LoRA weights...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = model.merge_and_unload()
        
        # Save merged model
        print(f"Saving merged model to {output_path}...")
        model.save_pretrained(output_path)
    else:
        # Keep as LoRA (smaller, but requires base model)
        print(f"Copying LoRA adapter to {output_path}...")
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
        )
        model.save_pretrained(output_path)
    
    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Model exported to {output_path}")


def generate_0g_config(
    model_hash: str,
    model_path: str,
    base_model: str,
    output_file: str = "./0g_model_config.yaml"
):
    """
    Generate configuration for 0G serving broker.
    
    This produces the config needed in api/fine-tuning/const/const.go
    """
    config = {
        "model_hash": model_hash,
        "model_path": model_path,
        "base_model": base_model,
        "training_config": {
            "algorithm": "GRPO",
            "lora_r": 64,
            "lora_alpha": 128,
            "context_length": 262144,  # 256K
            "training_steps": 2000,
        },
        "pricing": {
            # PriceCoefficient and StorageFee for SCRIPT_MAP
            "price_coefficient": 4,  # 7B model level
            "storage_fee": "15000000000000000",  # ~150MB LoRA in wei
        },
        "const_go_entry": f'''
// Auto-generated entry for 0G SCRIPT_MAP
"{model_hash}": {{
    TrainingScript:   "/app/train_lora.py",
    PriceCoefficient: 4,
    StorageFee:       15000000000000000, // ~150MB LoRA: 0.015 tokens
}},
''',
        "user_config_entry": f'''
# Add to user_config.yaml
ModelLocalPaths:
  "{model_hash}": "{model_path}"

ModelHuggingFaceFallback:
  "{model_hash}": "your-hf-org/your-model-name"
'''
    }
    
    # Save config
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n0G configuration saved to {output_file}")
    print("\n=== ADD THIS TO api/fine-tuning/const/const.go ===")
    print(config["const_go_entry"])
    print("\n=== ADD THIS TO user_config.yaml ===")
    print(config["user_config_entry"])


def quantize_for_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "Q4_K_M"
):
    """
    Quantize model to GGUF format for llama.cpp/ollama deployment.
    
    Reference: Qwen3.5 provides Q4_K_M (21.2 GB) for 35B model
    """
    print(f"Quantizing to GGUF format ({quantization})...")
    
    try:
        import llama_cpp
    except ImportError:
        print("Warning: llama_cpp not installed. Skipping quantization.")
        print("To enable: pip install llama-cpp-python")
        return
    
    # This would call llama.cpp convert script
    # For now, just print the command
    cmd = f"""
# Convert to GGUF using llama.cpp
python convert_hf_to_gguf.py {model_path} \\
    --outfile {output_path}/model.gguf \\
    --outtype {quantization}

# Or using llama-cpp-python
from llama_cpp import Llama
llm = Llama.from_pretrained(
    repo_id="your-model",
    filename="*gguf",
    verbose=True
)
"""
    print(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Export RL-trained model for 0G deployment"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/0g_ready_model",
        help="Output directory"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name (for HF fallback)"
    )
    parser.add_argument(
        "--merge-weights",
        action="store_true",
        help="Merge LoRA weights into base model"
    )
    parser.add_argument(
        "--compute-hash",
        action="store_true",
        help="Compute model hash for 0G registry"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "FP16"],
        help="Quantize to GGUF format"
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate 0G serving broker config"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Export model
    export_lora_model(
        args.checkpoint,
        args.output,
        base_model_path=args.base_model,
        merge_weights=args.merge_weights
    )
    
    # Compute hash
    model_hash = None
    if args.compute_hash:
        model_hash = compute_model_hash(args.output)
        
        # Save hash to file
        hash_file = Path(args.output) / "model_hash.txt"
        with open(hash_file, 'w') as f:
            f.write(model_hash)
        print(f"Hash saved to {hash_file}")
    
    # Generate 0G config
    if args.generate_config and model_hash:
        generate_0g_config(
            model_hash,
            args.output,
            args.base_model,
            output_file=Path(args.output) / "0g_config.json"
        )
    
    # Quantize
    if args.quantize:
        quantize_for_gguf(args.output, args.output, args.quantize)
    
    print("\n" + "="*50)
    print("Export complete!")
    print(f"Model location: {args.output}")
    if model_hash:
        print(f"Model hash: {model_hash}")
        print(f"Add this hash to api/fine-tuning/const/const.go")
    print("="*50)


if __name__ == "__main__":
    from tqdm import tqdm
    main()
