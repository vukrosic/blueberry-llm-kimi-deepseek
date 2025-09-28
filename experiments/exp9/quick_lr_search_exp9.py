#!/usr/bin/env python3
"""
Quick Learning Rate Search for Experiment 9
Runs a focused learning rate search with fewer steps for quick testing
"""

import torch
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from experiments.exp9.lr_search_exp9 import LearningRateSearchTrainer


def run_quick_lr_search():
    """Run quick learning rate search for testing"""
    print("🚀 Quick Learning Rate Search for Experiment 9")
    print("=" * 60)
    
    # Create configuration for quick learning rate search
    base_config = MoEModelConfig(
        max_steps=300,   # Very short training for quick search
        batch_size=128,
        max_tokens=100000,
        eval_every=30,   # More frequent evaluation
        num_documents=1000,
        max_seq_len=256,
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        num_experts=8,
        expert_top_k=2,
    )
    
    print(f"📋 Quick Search Configuration:")
    print(f"   Model: Attention+MLP 512d")
    print(f"   Batch Size: {base_config.batch_size}")
    print(f"   Model: {base_config.d_model}d, {base_config.n_layers}L, {base_config.n_heads}H")
    print(f"   Max Steps per LR: {base_config.max_steps}")
    
    # Define learning rates to test (focused range)
    learning_rates = [
        1e-4,   # Medium-low
        3e-4,   # Medium
        1e-3,   # Medium-high
        3e-3,   # High
    ]
    
    print(f"📋 Learning rates to test: {learning_rates}")
    
    # Create LR search trainer
    lr_trainer = LearningRateSearchTrainer(base_config, output_dir="exp9_quick_lr_search")
    
    # Run learning rate search
    print(f"\n🧪 Running quick learning rate search...")
    start_time = time.time()
    
    lr_results = lr_trainer.run_lr_search(
        learning_rates=learning_rates,
        max_steps=300,               # 300 steps per LR for quick testing
        eval_every=30,               # Evaluate every 30 steps
        early_stopping_patience=3    # Early stop after 3 evaluations without improvement
    )
    
    search_time = time.time() - start_time
    
    print(f"\n✅ Quick Learning Rate Search completed in {search_time/60:.2f} minutes!")
    
    # Find best learning rate
    valid_results = {k: v for k, v in lr_results.items() if 'error' not in v}
    if valid_results:
        best_lr_key = min(valid_results.keys(), key=lambda k: valid_results[k]['val_loss'])
        best_result = valid_results[best_lr_key]
        best_lr = best_result['learning_rate']
        
        print(f"\n🏆 Best Learning Rate Found:")
        print(f"   Learning Rate: {best_lr:.2e}")
        print(f"   Validation Loss: {best_result['val_loss']:.6f}")
        print(f"   Validation Accuracy: {best_result['val_accuracy']:.6f}")
        print(f"   Validation Perplexity: {best_result['val_perplexity']:.4f}")
        print(f"   Training Steps: {best_result['total_steps']}")
        print(f"   Training Time: {best_result['training_time_minutes']:.2f} min")
        
        # Save best LR recommendation
        recommendation = {
            'best_learning_rate': best_lr,
            'best_result': best_result,
            'all_results': lr_results,
            'search_time_minutes': search_time / 60,
            'recommendation': f"Use learning rate {best_lr:.2e} for long-term training"
        }
        
        recommendation_file = Path("exp9_quick_lr_search") / "lr_recommendation.json"
        with open(recommendation_file, 'w') as f:
            json.dump(recommendation, f, indent=2)
        
        print(f"\n💾 Recommendation saved to: {recommendation_file}")
        
        return best_lr, lr_results
    else:
        print("❌ No valid learning rate results found!")
        return None, lr_results


def main():
    """Main function to run quick learning rate search"""
    print("🚀 Quick Learning Rate Search for Experiment 9")
    print("=" * 60)
    
    try:
        best_lr, lr_results = run_quick_lr_search()
        
        if best_lr is None:
            print("❌ Quick learning rate search failed!")
            return False
        
        print(f"\n✅ Quick Learning Rate Search completed successfully!")
        print(f"📁 Results saved in: exp9_quick_lr_search/")
        print(f"🎯 Recommended LR for long-term training: {best_lr:.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in quick learning rate search: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
