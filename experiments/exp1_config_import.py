"""
Configuration for Experiment 1: DeepSeek Attention Integration (Using Original Implementation)

This configuration creates DeepSeek configs that use the original attention components.
"""

from typing import Dict, Any
from configuration_deepseek import DeepseekV3Config


def get_experiment_configs():
    """Get different configuration variants for the experiment using original DeepSeek components"""
    
    # Baseline config (standard attention, no DeepSeek features) - reduced size
    baseline_config = DeepseekV3Config(
        hidden_size=512,  # Reduced to match base config
        num_attention_heads=8,  # Reduced to match base config
        num_hidden_layers=6,  # Reduced to match base config
        intermediate_size=2048,  # Reduced to match base config
        vocab_size=1000,  # Will be set during training
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_rope_head_dim=None,  # Will use default
        v_head_dim=None,  # Will use default
        attention_bias=False,
        _attn_implementation="eager",
        rope_scaling=None
    )
    
    # DeepSeek config with LoRA projections - reduced size
    deepseek_lora_config = DeepseekV3Config(
        hidden_size=512,  # Reduced to match base config
        num_attention_heads=8,  # Reduced to match base config
        num_hidden_layers=6,  # Reduced to match base config
        intermediate_size=2048,  # Reduced to match base config
        vocab_size=1000,  # Will be set during training
        q_lora_rank=64,  # Reduced from 128
        kv_lora_rank=128,  # Reduced from 256
        qk_rope_head_dim=None,  # Will use default
        v_head_dim=None,  # Will use default
        attention_bias=False,
        _attn_implementation="eager",
        rope_scaling=None
    )
    
    # DeepSeek config with enhanced features - reduced size
    deepseek_enhanced_config = DeepseekV3Config(
        hidden_size=512,  # Reduced to match base config
        num_attention_heads=8,  # Reduced to match base config
        num_hidden_layers=6,  # Reduced to match base config
        intermediate_size=2048,  # Reduced to match base config
        vocab_size=1000,  # Will be set during training
        q_lora_rank=64,  # Reduced from 128
        kv_lora_rank=128,  # Reduced from 256
        qk_rope_head_dim=64,  # Reduced from 128
        v_head_dim=96,        # Reduced from 192
        attention_bias=True,
        _attn_implementation="eager",
        rope_scaling={
            "type": "linear",
            "factor": 2.0
        }
    )
    
    return {
        "baseline": baseline_config,
        "lora": deepseek_lora_config,
        "enhanced": deepseek_enhanced_config
    }


def create_config_from_moe_config(moe_config, variant="baseline"):
    """Create DeepSeek config from MoE config"""
    configs = get_experiment_configs()
    config = configs[variant]
    
    # Update with MoE config values
    config.hidden_size = moe_config.d_model
    config.num_attention_heads = moe_config.n_heads
    config.num_hidden_layers = moe_config.n_layers
    config.intermediate_size = moe_config.d_ff
    config.vocab_size = moe_config.vocab_size
    
    return config
