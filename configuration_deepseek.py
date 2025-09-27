"""
DeepSeek V3 Configuration
Minimal configuration class for DeepSeek attention components
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DeepseekV3Config:
    """Configuration for DeepSeek V3 model components"""
    
    # Basic model parameters
    hidden_size: int = 384
    num_attention_heads: int = 8
    num_hidden_layers: int = 6
    intermediate_size: int = 1536
    vocab_size: int = 1000
    
    # Attention parameters
    attention_dropout: float = 0.1
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    
    # DeepSeek specific attention parameters
    q_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    kv_lora_rank: Optional[int] = 64
    v_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    attention_bias: bool = False
    
    # RoPE scaling
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Attention implementation
    _attn_implementation: str = "eager"  # "eager" or "flash_attention_2"
    
    # MoE parameters
    n_routed_experts: Optional[int] = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: Optional[int] = None
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    routed_scaling_factor: float = 1.0
    scoring_func: str = "sigmoid"
    n_shared_experts: Optional[int] = None
    seq_aux: bool = True
    topk_method: str = "noaux_tc"
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = False
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Activation function
    hidden_act: str = "silu"
    
    # Other parameters
    initializer_range: float = 0.02
    use_cache: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = True
    
    def __post_init__(self):
        # Set default values based on other parameters
        if self.qk_rope_head_dim is None:
            self.qk_rope_head_dim = self.hidden_size // self.num_attention_heads
        if self.v_head_dim is None:
            self.v_head_dim = self.hidden_size // self.num_attention_heads
        if self.qk_nope_head_dim is None:
            self.qk_nope_head_dim = 0  # All head dim goes to RoPE
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size
        if self.kv_lora_rank is None:
            self.kv_lora_rank = self.hidden_size // self.num_attention_heads
