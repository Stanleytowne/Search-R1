#!/usr/bin/env python3

import torch
import gc
import argparse
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path: str, device: str):
    """Load model using transformers"""
    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    print(f"Model loaded successfully")
    return model

def subtract_models(model_after, model_before):
    """Calculate weight differences between two models"""
    print("Calculating weight differences...")
    deltas = {}
    
    # Optimization: Convert before_model params to a dict once to avoid repeated lookup overhead
    print("  Indexing base model parameters...")
    params_before = dict(model_before.named_parameters())
    
    # Iterate through all parameters
    count = 0
    for name, param_after in model_after.named_parameters():
        if name in params_before:
            param_before = params_before[name]
            # Ensure they are on same device for calculation if needed, 
            # though usually data subtraction handles it if both are loaded similarly.
            deltas[name] = param_after.data - param_before.data
            count += 1
    
    print(f"Calculated {count} weight deltas")
    return deltas

def apply_deltas_to_model(target_model, deltas):
    """Apply deltas to target model"""
    print("Applying deltas...")
    applied_count = 0
    with torch.no_grad():
        for name, param in target_model.named_parameters():
            if name in deltas:
                # Ensure delta is on the same device as the target parameter
                diff = deltas[name].to(param.device)
                param.data.add_(diff)
                applied_count += 1
    print(f"Applied deltas to {applied_count} parameters")

def clear_memory():
    """Helper to clean up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(
        description="Calculate GRPO weight deltas and apply to MULTIPLE target models"
    )
    
    parser.add_argument('--grpo_model', type=str, required=True,
                       help='Path to GRPO-trained model')
    parser.add_argument('--base_model', type=str, required=True,
                       help='Path to base model before GRPO training')
    
    # Modified to accept multiple arguments
    parser.add_argument('--target_models', type=str, nargs='+', required=True,
                       help='List of paths to target models to apply deltas')
    parser.add_argument('--output_dirs', type=str, nargs='+', required=True,
                       help='List of output paths corresponding to target models')
    
    args = parser.parse_args()
    
    # Validation
    if len(args.target_models) != len(args.output_dirs):
        print(f"Error: Number of target models ({len(args.target_models)}) "
              f"does not match number of output directories ({len(args.output_dirs)})")
        sys.exit(1)

    print("="*80)
    print("GRPO Weight Inheritance (Multi-Target)")
    print("="*80)
    
    # Step 1: Load models for Delta Calculation
    print("\n[Phase 1] Calculating Deltas")
    print("Loading GRPO model...")
    grpo_model = load_model(args.grpo_model, 'cpu')
    
    print("Loading base model...")
    base_model = load_model(args.base_model, 'cpu')
    
    # Step 2: Calculate deltas
    deltas = subtract_models(grpo_model, base_model)

    # Free memory from grpo and base models
    print("\nReleasing GRPO and base models from memory...")
    del grpo_model
    del base_model
    clear_memory()
    
    # Step 3: Iterate through targets
    total_targets = len(args.target_models)
    print(f"\n[Phase 2] Applying to {total_targets} target models")
    
    for i, (tgt_path, out_path) in enumerate(zip(args.target_models, args.output_dirs)):
        print(f"\n--- Processing Target {i+1}/{total_targets} ---")
        print(f"Input:  {tgt_path}")
        print(f"Output: {out_path}")
        
        # Load Target
        try:
            target_model = load_model(tgt_path, 'cpu')
            
            # Apply
            apply_deltas_to_model(target_model, deltas)
            
            # Save Model
            print(f"Saving updated model to: {out_path}")
            Path(out_path).mkdir(parents=True, exist_ok=True)
            target_model.save_pretrained(out_path)
            
            # Handle Tokenizer (Load from target input, save to output)
            print("Saving tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(tgt_path)
            tokenizer.save_pretrained(out_path)
            
        except Exception as e:
            print(f"ERROR processing target {tgt_path}: {e}")
        finally:
            # Cleanup specific target model
            if 'target_model' in locals():
                del target_model
            clear_memory()

    print("\n" + "="*80)
    print("✓ All tasks completed!")
    print("="*80)

if __name__ == '__main__':
    main()