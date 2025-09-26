#!/usr/bin/env python3
"""
Test a single configuration to debug the CUDA errors
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.exp2_config_import import get_architecture_search_configs, create_moe_config_from_architecture
from experiments.exp2_architecture_search_trainer import ArchitectureSearchTrainer
from data.loader import load_and_cache_data

def test_single_config():
    """Test a single configuration"""
    
    print("🔍 Testing single configuration...")
    
    # Create trainer
    trainer = ArchitectureSearchTrainer()
    
    # Get a single config
    all_configs = get_architecture_search_configs()
    config_name = "medium_baseline"
    config = all_configs[config_name]
    
    print(f"📋 Testing: {config_name}")
    print(f"   Config: {config}")
    
    # Create MoE config
    moe_config = create_moe_config_from_architecture("medium", "fast")
    moe_config.vocab_size = trainer.vocab_size
    
    print(f"   MoE Config: {moe_config}")
    print(f"   Vocab size: {moe_config.vocab_size}")
    
    # Test model creation
    try:
        model = trainer.create_model(config, config_name, moe_config)
        print(f"✅ Model created successfully")
        print(f"   Parameters: {trainer.count_parameters(model):,}")
        
        # Test forward pass
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Create dummy input
        batch_size = 2
        seq_len = 128
        dummy_input = torch.randint(0, moe_config.vocab_size, (batch_size, seq_len))
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
        
        print(f"   Testing forward pass with input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            if "baseline" in config_name:
                output = model(dummy_input, return_aux_loss=False)
            else:
                output = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_config()
