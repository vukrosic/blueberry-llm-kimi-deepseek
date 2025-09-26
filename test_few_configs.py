#!/usr/bin/env python3
"""
Test a few configurations to verify the fix works
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.exp2_config_import import get_architecture_search_configs, create_moe_config_from_architecture
from experiments.exp2_architecture_search_trainer import ArchitectureSearchTrainer

def test_few_configs():
    """Test a few configurations"""
    
    print("🔍 Testing a few configurations...")
    
    # Create trainer
    trainer = ArchitectureSearchTrainer()
    
    # Get a few configs to test
    all_configs = get_architecture_search_configs()
    test_configs = ["tiny_baseline", "medium_lora_small", "large_enhanced_medium"]
    
    for config_name in test_configs:
        print(f"\n📋 Testing: {config_name}")
        
        try:
            config = all_configs[config_name]
            
            # Extract model size from name
            size_name = config_name.split('_')[0]
            
            # Create MoE config
            moe_config = create_moe_config_from_architecture(size_name, "fast")
            moe_config.vocab_size = trainer.vocab_size
            
            print(f"   Model: {moe_config.d_model}d, {moe_config.n_layers}L, {moe_config.n_heads}H")
            
            # Train configuration
            result = trainer.train_configuration(config, config_name, moe_config, "fast")
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Success: Loss={result['val_loss']:.4f}, Time={result['training_time_minutes']:.1f}min")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_few_configs()
