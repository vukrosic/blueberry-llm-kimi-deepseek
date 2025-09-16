#!/usr/bin/env python3
"""
Test script for T4 Speedrun Challenge

This script tests the speedrun functionality without actually running training.
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from speedrun.config import (
    T4SpeedrunConfig,
    get_t4_speedrun_config,
    create_custom_t4_config,
    get_memory_optimized_config,
    get_performance_optimized_config,
    get_balanced_config
)
from speedrun.speedrun import SpeedrunTimer, SpeedrunValidator
from speedrun.leaderboard import SpeedrunLeaderboard


def test_configurations():
    """Test all configuration presets."""
    print("🧪 Testing Speedrun Configurations")
    print("="*50)
    
    configs = [
        ("Default", get_t4_speedrun_config),
        ("Memory Optimized", get_memory_optimized_config),
        ("Performance Optimized", get_performance_optimized_config),
        ("Balanced", get_balanced_config),
    ]
    
    for name, config_func in configs:
        print(f"\n📋 Testing {name} Configuration:")
        try:
            config = config_func()
            
            # Validate constraints
            if config.validate_speedrun_constraints():
                print(f"  ✅ Constraints validated")
            else:
                print(f"  ❌ Constraints failed")
            
            # Estimate memory
            memory_gb = config.estimate_memory_usage()
            print(f"  📊 Memory usage: {memory_gb:.2f} GB")
            
            # Print key parameters
            print(f"  🤖 Model: {config.d_model}d-{config.n_layers}L-{config.n_heads}H")
            print(f"  📦 Batch: {config.batch_size} (accumulation: {config.gradient_accumulation_steps})")
            print(f"  🧠 Experts: {config.num_experts}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")


def test_custom_configuration():
    """Test custom configuration creation."""
    print("\n🧪 Testing Custom Configuration")
    print("="*50)
    
    try:
        config = create_custom_t4_config(
            d_model=320,
            n_heads=8,
            n_layers=6,
            batch_size=20,
            muon_lr=0.008,
            max_steps=2500
        )
        
        print("✅ Custom configuration created successfully")
        print(f"  Model: {config.d_model}d-{config.n_layers}L-{config.n_heads}H")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.muon_lr}")
        print(f"  Max steps: {config.max_steps}")
        
        # Validate constraints
        if config.validate_speedrun_constraints():
            print("  ✅ Constraints validated")
        else:
            print("  ❌ Constraints failed")
            
    except Exception as e:
        print(f"❌ Custom configuration failed: {e}")


def test_timer():
    """Test speedrun timer functionality."""
    print("\n🧪 Testing Speedrun Timer")
    print("="*50)
    
    try:
        timer = SpeedrunTimer(5)  # 5 minute limit for testing
        
        print("✅ Timer created successfully")
        print(f"  Time limit: {timer.time_limit_seconds} seconds")
        
        # Test time checking
        if timer.check_time_limit():
            print("  ✅ Time limit check passed")
        else:
            print("  ❌ Time limit check failed")
        
        # Test remaining time
        remaining = timer.get_remaining_time()
        print(f"  ⏱️ Remaining time: {remaining:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Timer test failed: {e}")


def test_validator():
    """Test speedrun validator."""
    print("\n🧪 Testing Speedrun Validator")
    print("="*50)
    
    try:
        config = get_t4_speedrun_config()
        validator = SpeedrunValidator(config)
        
        print("✅ Validator created successfully")
        
        # Test hardware validation (may fail if no GPU)
        try:
            if validator.validate_hardware():
                print("  ✅ Hardware validation passed")
            else:
                print("  ⚠️ Hardware validation failed (expected if no GPU)")
        except Exception as e:
            print(f"  ⚠️ Hardware validation error: {e}")
        
        # Test config validation
        if validator.validate_config():
            print("  ✅ Configuration validation passed")
        else:
            print("  ❌ Configuration validation failed")
            
    except Exception as e:
        print(f"❌ Validator test failed: {e}")


def test_leaderboard():
    """Test leaderboard functionality."""
    print("\n🧪 Testing Leaderboard")
    print("="*50)
    
    try:
        leaderboard = SpeedrunLeaderboard("test_leaderboard.json")
        
        print("✅ Leaderboard created successfully")
        
        # Test adding a mock entry
        mock_results = {
            'timestamp': '2024-01-01T00:00:00',
            'completed': True,
            'final_val_loss': 2.345678,
            'final_val_accuracy': 0.4567,
            'final_val_perplexity': 10.45,
            'total_time_minutes': 28.5,
            'final_step': 2400,
            'steps_per_second': 1.4,
            'memory_usage_gb': 12.3,
            'time_exceeded': False,
        }
        
        if leaderboard.add_entry(mock_results, "TestParticipant"):
            print("  ✅ Mock entry added successfully")
        else:
            print("  ❌ Failed to add mock entry")
        
        # Test leaderboard display
        leaderboard.print_leaderboard(5)
        
        # Test statistics
        leaderboard.print_statistics()
        
        # Clean up test leaderboard
        leaderboard.clear_leaderboard()
        print("  🗑️ Test leaderboard cleared")
        
    except Exception as e:
        print(f"❌ Leaderboard test failed: {e}")


def main():
    """Run all tests."""
    print("🚀 T4 Speedrun Challenge - Test Suite")
    print("="*60)
    
    test_configurations()
    test_custom_configuration()
    test_timer()
    test_validator()
    test_leaderboard()
    
    print("\n" + "="*60)
    print("🎉 All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
