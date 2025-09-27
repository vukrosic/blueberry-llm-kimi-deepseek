#!/usr/bin/env python3
"""
Quick test script for Experiment 8
Tests that all models can be created successfully
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from experiments.exp8.exp8_reduced_ablation_models import (
    create_reduced_ablation_model,
    REDUCED_ABLATION_MODELS,
    print_reduced_ablation_summary
)

def test_model_creation():
    """Test that all models can be created without errors"""
    print("🧪 Testing Experiment 8 Model Creation")
    print("=" * 50)
    
    # Create a minimal config for testing
    config = MoEModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=3,
        d_ff=2048,
        num_experts=8,
        expert_top_k=2,
        vocab_size=32000,
        max_seq_len=256,
        batch_size=16,
        max_steps=10,
        max_tokens=10000,
        num_documents=100,
    )
    
    successful = 0
    failed = 0
    
    for model_name in REDUCED_ABLATION_MODELS.keys():
        try:
            print(f"Testing {model_name}...", end=" ")
            model = create_reduced_ablation_model(model_name, config)
            print("✅")
            successful += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
    
    print(f"\n📊 Results:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📋 Total Models: {len(REDUCED_ABLATION_MODELS)}")
    
    if failed == 0:
        print("\n🎉 All models created successfully!")
        print_reduced_ablation_summary()
    else:
        print(f"\n⚠️  {failed} models failed to create")
    
    return failed == 0

if __name__ == "__main__":
    success = test_model_creation()
    sys.exit(0 if success else 1)
