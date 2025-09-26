"""
Configuration for Experiment 1: DeepSeek Attention Integration

This configuration extends the base MoE config with DeepSeek-specific attention parameters.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from configs.moe_config import MoEModelConfig


@dataclass
class DeepSeekMoEConfig(MoEModelConfig):
    """Extended MoE config with DeepSeek attention features"""
    
    # DeepSeek Attention Parameters
    q_lora_rank: Optional[int] = None  # LoRA rank for Q projection (None = standard)
    kv_lora_rank: Optional[int] = 64   # LoRA rank for KV projection
    qk_rope_head_dim: Optional[int] = None  # Head dim for QK RoPE (None = d_k)
    v_head_dim: Optional[int] = None   # Head dim for V (None = d_k)
    attention_bias: bool = False       # Whether to use bias in attention projections
    use_flash_attention: bool = False  # Whether to use Flash Attention 2
    
    # RoPE Scaling Configuration
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Experiment-specific parameters
    experiment_name: str = "deepseek_attention_integration"
    baseline_comparison: bool = True   # Whether to run baseline comparison
    
    def __post_init__(self):
        super().__post_init__()
        
        # Set default values for DeepSeek parameters
        if self.qk_rope_head_dim is None:
            self.qk_rope_head_dim = self.d_k
        if self.v_head_dim is None:
            self.v_head_dim = self.d_k
        if self.kv_lora_rank is None:
            self.kv_lora_rank = self.d_k // 2
        
        # Validate DeepSeek parameters
        assert self.qk_rope_head_dim <= self.d_k, "qk_rope_head_dim must be <= d_k"
        assert self.v_head_dim <= self.d_k, "v_head_dim must be <= d_k"
        if self.q_lora_rank is not None:
            assert self.q_lora_rank > 0, "q_lora_rank must be positive"
        assert self.kv_lora_rank > 0, "kv_lora_rank must be positive"


def get_experiment_configs():
    """Get different configuration variants for the experiment"""
    
    # Baseline config (standard attention)
    baseline_config = DeepSeekMoEConfig(
        experiment_name="baseline_standard_attention",
        q_lora_rank=None,
        kv_lora_rank=None,
        use_flash_attention=False,
        rope_scaling=None
    )
    
    # DeepSeek config with LoRA projections
    deepseek_lora_config = DeepSeekMoEConfig(
        experiment_name="deepseek_lora_projections",
        q_lora_rank=32,
        kv_lora_rank=64,
        use_flash_attention=False,
        rope_scaling=None
    )
    
    # DeepSeek config with Flash Attention
    deepseek_flash_config = DeepSeekMoEConfig(
        experiment_name="deepseek_flash_attention",
        q_lora_rank=32,
        kv_lora_rank=64,
        use_flash_attention=True,
        rope_scaling=None
    )
    
    # DeepSeek config with RoPE scaling
    deepseek_rope_config = DeepSeekMoEConfig(
        experiment_name="deepseek_rope_scaling",
        q_lora_rank=32,
        kv_lora_rank=64,
        use_flash_attention=True,
        rope_scaling={
            "type": "linear",
            "factor": 2.0
        }
    )
    
    # Full DeepSeek config (all features)
    deepseek_full_config = DeepSeekMoEConfig(
        experiment_name="deepseek_full_features",
        q_lora_rank=32,
        kv_lora_rank=64,
        qk_rope_head_dim=32,  # Smaller RoPE head dim
        v_head_dim=48,        # Larger V head dim
        attention_bias=True,
        use_flash_attention=True,
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
