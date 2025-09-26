"""
Configuration for Experiment 2: Fair Architecture Search (Fixed Model Size)

This configuration creates attention mechanism configurations with fixed model size
to test architectural differences in a fair comparison.
"""

from typing import Dict, Any, List, Tuple
from configuration_deepseek import DeepseekV3Config
from configs.moe_config import MoEModelConfig


def get_architecture_search_configs():
    """Get architecture search configurations with fixed model size to test attention mechanisms"""
    
    configs = {}
    
    # Fixed model size for fair comparison (medium size that works well)
    base_model_size = {
        "d_model": 512,
        "n_heads": 8, 
        "n_layers": 8,
        "d_ff": 2048
    }
    
    # Attention mechanisms to test (keeping model size constant)
    attention_configs = [
        ("baseline", {
            "q_lora_rank": None,
            "kv_lora_rank": None,
            "qk_rope_head_dim": None,
            "v_head_dim": None,
            "attention_bias": False,
            "rope_scaling": None
        }),
        ("lora_small", {
            "q_lora_rank": 32,
            "kv_lora_rank": 64,
            "qk_rope_head_dim": None,
            "v_head_dim": None,
            "attention_bias": False,
            "rope_scaling": None
        }),
        ("lora_medium", {
            "q_lora_rank": 64,
            "kv_lora_rank": 128,
            "qk_rope_head_dim": None,
            "v_head_dim": None,
            "attention_bias": False,
            "rope_scaling": None
        }),
        ("lora_large", {
            "q_lora_rank": 128,
            "kv_lora_rank": 256,
            "qk_rope_head_dim": None,
            "v_head_dim": None,
            "attention_bias": False,
            "rope_scaling": None
        }),
        ("enhanced_small", {
            "q_lora_rank": 32,
            "kv_lora_rank": 64,
            "qk_rope_head_dim": 32,
            "v_head_dim": 48,
            "attention_bias": True,
            "rope_scaling": {"type": "linear", "factor": 1.5}
        }),
        ("enhanced_medium", {
            "q_lora_rank": 64,
            "kv_lora_rank": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 96,
            "attention_bias": True,
            "rope_scaling": {"type": "linear", "factor": 2.0}
        }),
        ("enhanced_large", {
            "q_lora_rank": 128,
            "kv_lora_rank": 256,
            "qk_rope_head_dim": 128,
            "v_head_dim": 192,
            "attention_bias": True,
            "rope_scaling": {"type": "linear", "factor": 2.5}
        }),
        ("rope_only", {
            "q_lora_rank": None,
            "kv_lora_rank": None,
            "qk_rope_head_dim": 64,
            "v_head_dim": 96,
            "attention_bias": False,
            "rope_scaling": {"type": "linear", "factor": 2.0}
        }),
        ("bias_only", {
            "q_lora_rank": None,
            "kv_lora_rank": None,
            "qk_rope_head_dim": None,
            "v_head_dim": None,
            "attention_bias": True,
            "rope_scaling": None
        }),
        # Additional variants for more comprehensive testing
        ("lora_xl", {
            "q_lora_rank": 256,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": None,
            "v_head_dim": None,
            "attention_bias": False,
            "rope_scaling": None
        }),
        ("enhanced_xl", {
            "q_lora_rank": 256,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 128,
            "v_head_dim": 192,
            "attention_bias": True,
            "rope_scaling": {"type": "linear", "factor": 3.0}
        }),
        ("rope_small", {
            "q_lora_rank": None,
            "kv_lora_rank": None,
            "qk_rope_head_dim": 32,
            "v_head_dim": 48,
            "attention_bias": False,
            "rope_scaling": {"type": "linear", "factor": 1.5}
        }),
        ("rope_large", {
            "q_lora_rank": None,
            "kv_lora_rank": None,
            "qk_rope_head_dim": 128,
            "v_head_dim": 192,
            "attention_bias": False,
            "rope_scaling": {"type": "linear", "factor": 2.5}
        }),
    ]
    
    # Create configurations with fixed model size
    for attn_name, attn_params in attention_configs:
        config_name = f"medium_{attn_name}"  # All use medium size for fair comparison
        
        config = DeepseekV3Config(
            hidden_size=base_model_size["d_model"],
            num_attention_heads=base_model_size["n_heads"],
            num_hidden_layers=base_model_size["n_layers"],
            intermediate_size=base_model_size["d_ff"],
            vocab_size=49152,  # Will be set during training
            _attn_implementation="eager",
            **attn_params
        )
        
        configs[config_name] = config
    
    return configs


def get_training_configs():
    """Get training configurations for fair architecture search"""
    
    return {
        "fast": {
            "max_steps": 50,
            "batch_size": 16,
            "eval_every": 10,
            "description": "Fast training for architecture comparison"
        }
    }


def create_moe_config_from_architecture(size_name: str, training_mode: str = "fast") -> MoEModelConfig:
    """Create MoE config with fixed model size for fair comparison"""
    
    # Fixed model size for all configurations (fair comparison)
    fixed_model_size = {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,
        "d_ff": 2048
    }
    
    training_configs = get_training_configs()
    train_params = training_configs[training_mode]
    
    return MoEModelConfig(
        max_steps=train_params["max_steps"],
        batch_size=train_params["batch_size"],
        max_tokens=1000000,  # Reduced for faster training
        eval_every=train_params["eval_every"],
        num_documents=4000,  # Reduced for faster training
        max_seq_len=512,
        d_model=fixed_model_size["d_model"],
        n_heads=fixed_model_size["n_heads"],
        n_layers=fixed_model_size["n_layers"],
        d_ff=fixed_model_size["d_ff"],
    )


def get_experiment_configs():
    """Get experiment configurations for backward compatibility"""
    return get_architecture_search_configs()


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
