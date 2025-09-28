#!/usr/bin/env python3
"""
Test script for Experiment 9
Verifies that the trainer and inference scripts work correctly
"""

import sys
import os

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from experiments.exp9.exp9_trainer import LongTermExperiment9Trainer
from experiments.exp8.exp8_reduced_ablation_models import AttentionMLP_512dModel

def test_model_creation():
    """Test that the Attention+MLP model can be created"""
    print("🧪 Testing Experiment 9 Model Creation")
    print("=" * 50)
    
    # Create a minimal config for testing
    config = MoEModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=12,  # Deeper model (was 3, now 12)
        d_ff=2048,
        num_experts=8,
        expert_top_k=2,
        vocab_size=32000,
        max_seq_len=256,
        batch_size=16,
        max_steps=100,
        max_tokens=10000,
        num_documents=100,
    )
    
    try:
        print("Creating Attention+MLP 512d model...", end=" ")
        model = AttentionMLP_512dModel(config)
        print("✅")
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_trainer_creation():
    """Test that the trainer can be created"""
    print("\n🧪 Testing Experiment 9 Trainer Creation")
    print("=" * 50)
    
    try:
        # Create a minimal config
        config = MoEModelConfig(
            d_model=512,
            n_heads=8,
            n_layers=12,  # Deeper model (was 3, now 12)
            d_ff=2048,
            num_experts=8,
            expert_top_k=2,
            vocab_size=32000,
            max_seq_len=256,
            batch_size=16,
            max_steps=100,
            max_tokens=10000,
            num_documents=100,
        )
        
        print("Creating LongTermExperiment9Trainer...", end=" ")
        trainer = LongTermExperiment9Trainer(config, output_dir="test_exp9_results")
        print("✅")
        
        print(f"📁 Output directory: {trainer.output_dir}")
        print(f"📊 Dataset: {len(trainer.train_dataset)} train, {len(trainer.val_dataset)} val")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_short_training():
    """Test a very short training run"""
    print("\n🧪 Testing Short Training Run")
    print("=" * 50)
    
    try:
        # Create config for short test
        config = MoEModelConfig(
            d_model=512,
            n_heads=8,
            n_layers=12,  # Deeper model (was 3, now 12)
            d_ff=2048,
            num_experts=8,
            expert_top_k=2,
            vocab_size=32000,
            max_seq_len=256,
            batch_size=16,
            max_steps=100,
            max_tokens=10000,
            num_documents=100,
        )
        
        print("Creating trainer...", end=" ")
        trainer = LongTermExperiment9Trainer(config, output_dir="test_exp9_results")
        print("✅")
        
        print("Running short training (50 steps)...", end=" ")
        results = trainer.run_long_term_training(
            total_steps=50,
            checkpoint_every=25,
            eval_every=10,
            hellaswag_every=50  # Skip HellaSwag for short test
        )
        print("✅")
        
        if "error" not in results:
            print(f"📊 Final loss: {results['val_loss']:.6f}")
            print(f"📊 Final accuracy: {results['val_accuracy']:.4f}")
            print(f"⏱️ Training time: {results['training_time_minutes']:.2f} min")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Experiment 9 Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Trainer Creation", test_trainer_creation),
        ("Short Training", test_short_training),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Experiment 9 is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Run: python exp9_trainer.py")
        print("   2. Run: python exp9_inference.py --mode interactive")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
