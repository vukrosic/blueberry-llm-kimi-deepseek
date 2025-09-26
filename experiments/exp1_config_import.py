"""
Configuration for Experiment 1: DeepSeek Attention Integration (Using Original Implementation)

This configuration creates DeepSeek configs that use the original attention components.
"""

from typing import Dict, Any
from configuration_deepseek import DeepseekV3Config


def get_experiment_configs():
    """Get different configuration variants for the experiment using original DeepSeek components"""
    
    # Baseline config (standard attention, no DeepSeek features)
    baseline_config = DeepseekV3Config(
        hidden_size=384,
        num_attention_heads=8,
        num_hidden_layers=6,
        intermediate_size=1536,
        vocab_size=1000,  # Will be set during training
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_rope_head_dim=None,  # Will use default
        v_head_dim=None,  # Will use default
        attention_bias=False,
        _attn_implementation="eager",
        rope_scaling=None
    )
    
    # DeepSeek config with LoRA projections
    deepseek_lora_config = DeepseekV3Config(
        hidden_size=384,
        num_attention_heads=8,
        num_hidden_layers=6,
        intermediate_size=1536,
        vocab_size=1000,  # Will be set during training
        q_lora_rank=32,
        kv_lora_rank=64,
        qk_rope_head_dim=None,  # Will use default
        v_head_dim=None,  # Will use default
        attention_bias=False,
        _attn_implementation="eager",
        rope_scaling=None
    )
    
    # DeepSeek config with Flash Attention
    deepseek_flash_config = DeepseekV3Config(
        hidden_size=384,
        num_attention_heads=8,
        num_hidden_layers=6,
        intermediate_size=1536,
        vocab_size=1000,  # Will be set during training
        q_lora_rank=32,
        kv_lora_rank=64,
        qk_rope_head_dim=None,  # Will use default
        v_head_dim=None,  # Will use default
        attention_bias=False,
        _attn_implementation="flash_attention_2",
        rope_scaling=None
    )
    
    # DeepSeek config with RoPE scaling
    deepseek_rope_config = DeepseekV3Config(
        hidden_size=384,
        num_attention_heads=8,
        num_hidden_layers=6,
        intermediate_size=1536,
        vocab_size=1000,  # Will be set during training
        q_lora_rank=32,
        kv_lora_rank=64,
        qk_rope_head_dim=None,  # Will use default
        v_head_dim=None,  # Will use default
        attention_bias=False,
        _attn_implementation="flash_attention_2",
        rope_scaling={
            "type": "linear",
            "factor": 2.0
        }
    )
    
    # Full DeepSeek config (all features)
    deepseek_full_config = DeepseekV3Config(
        hidden_size=384,
        num_attention_heads=8,
        num_hidden_layers=6,
        intermediate_size=1536,
        vocab_size=1000,  # Will be set during training
        q_lora_rank=32,
        kv_lora_rank=64,
        qk_rope_head_dim=32,  # Smaller RoPE head dim
        v_head_dim=48,        # Larger V head dim
        attention_bias=True,
        _attn_implementation="flash_attention_2",
        rope_scaling={
            "type": "dynamic",
            "factor": 1.5
        }
    )
    
    return {
        "baseline": baseline_config,
        "lora": deepseek_lora_config,
        "flash": deepseek_flash_config,
        "rope": deepseek_rope_config,
        "full": deepseek_full_config
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
