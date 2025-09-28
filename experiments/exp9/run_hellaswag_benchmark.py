#!/usr/bin/env python3
"""
Run HellaSwag benchmark on Experiment 9 latest checkpoint
Loads the latest checkpoint and evaluates it on HellaSwag benchmark
"""

import torch
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from experiments.exp8.exp8_reduced_ablation_models import AttentionMLP_512dModel
from benchmark_evaluator import HellaSwagEvaluator


def load_latest_checkpoint(checkpoint_dir: str = "exp9_results/checkpoints"):
    """Load the latest checkpoint from experiment 9"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_path.glob("checkpoint_step_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
    
    # Sort by step number and get the latest
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    latest_checkpoint = checkpoint_files[-1]
    
    print(f"🔍 Found {len(checkpoint_files)} checkpoints")
    print(f"📁 Latest checkpoint: {latest_checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    step = checkpoint['step']
    
    print(f"✅ Loaded checkpoint from step {step}")
    print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
    print(f"   Timestamp: {time.ctime(checkpoint['timestamp'])}")
    
    return checkpoint, step


def create_model_from_checkpoint(checkpoint: Dict[str, Any], step: int):
    """Create model and load checkpoint state"""
    print(f"\n🧪 Creating Attention+MLP 512d model...")
    
    # Load tokenizer first to get vocab_size
    temp_config = MoEModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        num_experts=8,
        expert_top_k=2,
        vocab_size=32000,
        max_seq_len=256,
        batch_size=1,
        max_steps=100,
        max_tokens=10000,
        num_documents=100,
    )
    
    # Load tokenizer to get actual vocab size
    _, tokenizer, _ = load_and_cache_data(temp_config)
    vocab_size = temp_config.vocab_size
    
    # Create model config
    config = MoEModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        num_experts=8,
        expert_top_k=2,
        vocab_size=vocab_size,
        max_seq_len=256,
        batch_size=1,
        max_steps=100,
        max_tokens=10000,
        num_documents=100,
    )
    
    # Create model
    model = AttentionMLP_512dModel(config)
    
    # Load checkpoint state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model created and loaded from step {step}")
    print(f"   Device: {device}")
    print(f"   Vocab size: {vocab_size}")
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    return model, tokenizer, config


def run_hellaswag_benchmark(model, tokenizer, config, model_name: str = "attention_mlp_512d"):
    """Run HellaSwag benchmark on the loaded model"""
    print(f"\n🧪 Running HellaSwag benchmark on {model_name}...")
    
    # Create HellaSwag evaluator
    evaluator = HellaSwagEvaluator(output_dir="exp9_results/hellaswag_benchmark")
    
    # Run evaluation
    start_time = time.time()
    results = evaluator.evaluate_model(model, model_name, tokenizer)
    evaluation_time = time.time() - start_time
    
    # Add evaluation time to results
    results['evaluation_time_seconds'] = evaluation_time
    
    print(f"✅ HellaSwag benchmark completed in {evaluation_time:.2f} seconds")
    print(f"📊 Results: {results}")
    
    return results


def update_results_file(results: Dict[str, Any], results_file: str = "exp9_results/hellaswag_benchmark/attention_mlp_512d_hellaswag_results.json"):
    """Update the results JSON file with benchmark results"""
    results_path = Path(results_file)
    
    # Ensure directory exists
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}")


def main():
    """Main function to run HellaSwag benchmark on latest checkpoint"""
    print("🚀 HellaSwag Benchmark for Experiment 9")
    print("=" * 60)
    
    try:
        # Load latest checkpoint
        checkpoint, step = load_latest_checkpoint()
        
        # Create model from checkpoint
        model, tokenizer, config = create_model_from_checkpoint(checkpoint, step)
        
        # Run HellaSwag benchmark
        results = run_hellaswag_benchmark(model, tokenizer, config)
        
        # Update results file
        update_results_file(results)
        
        print(f"\n✅ HellaSwag benchmark completed successfully!")
        print(f"📊 Final Results:")
        print(f"   Model: attention_mlp_512d (step {step})")
        print(f"   Accuracy: {results.get('accuracy', 'N/A')}")
        print(f"   F1: {results.get('f1', 'N/A')}")
        print(f"   Exact Match: {results.get('exact_match', 'N/A')}")
        print(f"   Evaluation Time: {results.get('evaluation_time_seconds', 'N/A')} seconds")
        print(f"   Status: {results.get('status', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error running HellaSwag benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
