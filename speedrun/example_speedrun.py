#!/usr/bin/env python3
"""
Example T4 Speedrun Implementation

This script demonstrates how to create custom speedrun configurations
and strategies for the T4 speedrun challenge.
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from speedrun.config import create_custom_t4_config
from speedrun.speedrun import run_speedrun


def example_memory_efficient_strategy():
    """Example: Memory-efficient strategy focusing on model depth."""
    print("🧠 Memory-Efficient Strategy Example")
    
    config = create_custom_t4_config(
        d_model=192,           # Smaller model dimension
        n_heads=6,            # Must divide d_model
        n_layers=8,           # More layers for capacity
        d_ff=768,             # Smaller FFN
        batch_size=8,         # Small batch size
        gradient_accumulation_steps=8,  # High accumulation
        max_seq_len=256,      # Shorter sequences
        num_experts=6,        # Fewer experts
        expert_top_k=2,       # Top-2 experts
        muon_lr=0.012,        # Slightly higher LR
        max_steps=3000,       # More steps
        eval_every=200,      # Frequent evaluation
    )
    
    print("Configuration:")
    print(f"  Model: {config.d_model}d-{config.n_layers}L-{config.n_heads}H")
    print(f"  Batch: {config.batch_size} (accumulation: {config.gradient_accumulation_steps})")
    print(f"  Sequence: {config.max_seq_len}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Learning Rate: {config.muon_lr}")
    
    return config


def example_performance_strategy():
    """Example: Performance-focused strategy with larger model."""
    print("🚀 Performance Strategy Example")
    
    config = create_custom_t4_config(
        d_model=384,           # Larger model
        n_heads=8,             # Standard heads
        n_layers=6,            # Moderate depth
        d_ff=1536,            # Larger FFN
        batch_size=24,         # Larger batch
        gradient_accumulation_steps=2,  # Low accumulation
        max_seq_len=512,       # Standard sequences
        num_experts=12,        # More experts
        expert_top_k=2,        # Top-2 experts
        muon_lr=0.008,         # Lower LR for stability
        max_steps=2000,        # Fewer steps
        eval_every=150,       # Frequent evaluation
    )
    
    print("Configuration:")
    print(f"  Model: {config.d_model}d-{config.n_layers}L-{config.n_heads}H")
    print(f"  Batch: {config.batch_size} (accumulation: {config.gradient_accumulation_steps})")
    print(f"  Sequence: {config.max_seq_len}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Learning Rate: {config.muon_lr}")
    
    return config


def example_balanced_strategy():
    """Example: Balanced strategy between memory and performance."""
    print("⚖️ Balanced Strategy Example")
    
    config = create_custom_t4_config(
        d_model=256,           # Medium model
        n_heads=8,             # Standard heads
        n_layers=7,            # Slightly deeper
        d_ff=1024,            # Standard FFN
        batch_size=16,         # Medium batch
        gradient_accumulation_steps=4,  # Medium accumulation
        max_seq_len=384,       # Medium sequences
        num_experts=8,         # Standard experts
        expert_top_k=2,        # Top-2 experts
        muon_lr=0.01,          # Standard LR
        max_steps=2500,        # Medium steps
        eval_every=180,       # Regular evaluation
    )
    
    print("Configuration:")
    print(f"  Model: {config.d_model}d-{config.n_layers}L-{config.n_heads}H")
    print(f"  Batch: {config.batch_size} (accumulation: {config.gradient_accumulation_steps})")
    print(f"  Sequence: {config.max_seq_len}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Learning Rate: {config.muon_lr}")
    
    return config


def run_example_speedrun(strategy_name: str = "balanced", time_limit: int = 30):
    """Run an example speedrun with the specified strategy."""
    
    print(f"🏃‍♂️ Running {strategy_name} speedrun strategy...")
    print(f"⏱️ Time limit: {time_limit} minutes")
    print("="*60)
    
    # Select strategy
    if strategy_name == "memory":
        config = example_memory_efficient_strategy()
    elif strategy_name == "performance":
        config = example_performance_strategy()
    elif strategy_name == "balanced":
        config = example_balanced_strategy()
    else:
        print(f"❌ Unknown strategy: {strategy_name}")
        print("Available strategies: memory, performance, balanced")
        return None
    
    # Estimate memory usage
    estimated_memory = config.estimate_memory_usage()
    
    # Run speedrun
    results = run_speedrun(config, time_limit)
    
    return results


def main():
    """Main entry point for example speedrun."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Example T4 Speedrun")
    parser.add_argument("--strategy", type=str, default="balanced",
                       choices=["memory", "performance", "balanced"],
                       help="Strategy to use")
    parser.add_argument("--time-limit", type=int, default=30,
                       help="Time limit in minutes")
    
    args = parser.parse_args()
    
    # Run example speedrun
    results = run_example_speedrun(args.strategy, args.time_limit)
    
    if results and results.results['completed']:
        print(f"\n🎉 Example speedrun completed!")
        print(f"Final validation loss: {results.results['final_val_loss']:.6f}")
        print(f"Time taken: {results.results['total_time_minutes']:.2f} minutes")
    else:
        print(f"\n❌ Example speedrun failed")
        if results:
            print(f"Error: {results.results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
