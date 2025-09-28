"""
Experiment 8: Reduced Ablation Study Models
Focused on 512 hidden dimension scale with powers of 2 ablations
Based on the provided JSON config architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from models.moe_llm import MoEMinimalLLM
from deepseek_modeling import (
    DeepseekV3RMSNorm, 
    DeepseekV3MLP, 
    DeepseekV3MoE,
    DeepseekV3Attention
)
from configuration_deepseek import DeepseekV3Config

# GLM4 MoE imports
try:
    from transformers import Glm4MoeConfig, Glm4MoeForCausalLM
    GLM4_MOE_AVAILABLE = True
except ImportError:
    GLM4_MOE_AVAILABLE = False
    print("Warning: GLM4 MoE not available, falling back to baseline MoE")


class DeepseekV3MLPWrapper(nn.Module):
    """Wrapper for DeepseekV3MLP to return (output, aux_loss) like MoE components"""
    
    def __init__(self, config):
        super().__init__()
        self.mlp = DeepseekV3MLP(config)
    
    def forward(self, x):
        output = self.mlp(x)
        return output, None  # No auxiliary loss for MLP


class DeepseekV3AttentionWrapper(nn.Module):
    """Wrapper for DeepseekV3Attention to return only the output tensor"""
    
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.attention = DeepseekV3Attention(config, layer_idx=layer_idx)
    
    def forward(self, x):
        # Create attention mask (all ones for causal attention)
        batch_size, seq_len = x.shape[:2]
        # Create 4D attention mask: (batch_size, 1, seq_len, seq_len)
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, device=x.device, dtype=torch.bool)
        
        output, _, _ = self.attention(x, attention_mask=attention_mask)  # Unpack tuple, ignore cache and attention weights
        return output


class GLM4MoEWrapper(nn.Module):
    """Wrapper for GLM4 MoE to return (output, aux_loss) like MoE components"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        if not GLM4_MOE_AVAILABLE:
            raise RuntimeError("GLM4 MoE not available")
        
        # Create a minimal GLM4 MoE config for just the MoE component
        head_dim = config.d_model // config.n_heads
        vocab_size = max(32000, config.vocab_size or 32000)  # Ensure minimum vocab size
        
        glm4_config = Glm4MoeConfig(
            vocab_size=vocab_size,
            hidden_size=config.d_model,
            num_hidden_layers=1,  # Single layer for this component
            num_attention_heads=config.n_heads,
            num_key_value_heads=config.n_heads,
            head_dim=head_dim,
            max_position_embeddings=max(2048, config.max_seq_len),
            attention_dropout=0.0,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.expert_top_k,
            moe_intermediate_size=max(1024, 4*config.d_model//2),
            n_shared_experts=1,
            use_cache=False,
            tie_word_embeddings=False,
            pad_token_id=min(151329, vocab_size - 1),  # Ensure padding_idx is within vocab_size
            eos_token_id=[min(151329, vocab_size - 1)],
            attention_bias=True,
        )
        
        # Create the model and extract just the MoE component
        self.model = Glm4MoeForCausalLM(glm4_config)
        # Get the MoE layer from the first transformer block
        self.moe_layer = self.model.model.layers[0].mlp
        
    def forward(self, x):
        # Use only the MoE component, bypassing attention and other components
        # The MoE layer expects normalized input
        output = self.moe_layer(x)
        # GLM4 MoE doesn't return auxiliary loss, so return None
        return output, None


def create_deepseek_config(moe_config: MoEModelConfig) -> DeepseekV3Config:
    """Create DeepSeek config from MoE config"""
    return DeepseekV3Config(
        hidden_size=moe_config.d_model,
        intermediate_size=moe_config.d_ff,
        num_attention_heads=moe_config.n_heads,
        num_hidden_layers=moe_config.n_layers,
        vocab_size=moe_config.vocab_size or 32000,
        max_position_embeddings=moe_config.max_seq_len,
        hidden_act="silu",
        attention_dropout=moe_config.dropout,
        # MoE specific
        n_routed_experts=moe_config.num_experts,
        num_experts_per_tok=moe_config.expert_top_k,
        moe_intermediate_size=moe_config.d_ff,
        routed_scaling_factor=1.0,
        # Attention specific
        q_lora_rank=128,
        kv_lora_rank=128,
        qk_rope_head_dim=moe_config.d_model // moe_config.n_heads,
        v_head_dim=moe_config.d_model // moe_config.n_heads,
        qk_nope_head_dim=0,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
    )


# =============================================================================
# REDUCED ABLATION MODELS - 512 SCALE FOCUSED
# =============================================================================

class BaselineModel(MoEMinimalLLM):
    """Baseline model - no DeepSeek components (control group)"""
    
    def __init__(self, config: MoEModelConfig):
        # Adjust for similar parameter count
        config.d_model = 512
        config.d_ff = 1024  # Smaller intermediate size for similar params
        super().__init__(config)
        print("🔧 Using Baseline Model (no DeepSeek components)")


class MLP_512dModel(MoEMinimalLLM):
    """DeepSeek MLP with 512d (matches target architecture scale)"""
    
    def __init__(self, config: MoEModelConfig):
        # Override to 512 dimensions
        config.d_model = 512
        config.d_ff = 2048  # Standard 4x d_model scaling
        super().__init__(config)
        
        # Replace MLP with DeepSeek MLP
        deepseek_config = create_deepseek_config(config)
        for i, block in enumerate(self.transformer_blocks):
            block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using MLP_512d Model (DeepSeek MLP: 512d)")


class AttentionMLP_512dModel(MoEMinimalLLM):
    """DeepSeek Attention + MLP with 512d (matches target architecture)"""
    
    def __init__(self, config: MoEModelConfig):
        # Override to 512 dimensions
        config.d_model = 512
        config.d_ff = 2048  # Standard 4x d_model scaling
        super().__init__(config)
        
        # Replace both attention and MLP
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using Attention+MLP_512d Model (DeepSeek Attention + MLP: 512d)")


class MoE_8e_2k_512dModel(MoEMinimalLLM):
    """GLM4 MoE with 8 experts, top-2, 512d (matches target architecture)"""
    
    def __init__(self, config: MoEModelConfig):
        # Override to 512 dimensions, 8 experts, top-2
        config.d_model = 512
        config.num_experts = 8
        config.expert_top_k = 2
        config.d_ff = 1024  # Smaller intermediate size for similar params
        super().__init__(config)
        
        # Replace MoE with GLM4 MoE
        if GLM4_MOE_AVAILABLE:
            for i, block in enumerate(self.transformer_blocks):
                block.feed_forward = GLM4MoEWrapper(config)
            print("🔧 Using MoE_8e_2k_512d Model (GLM4 MoE: 8 experts, top-2, 512d)")
        else:
            print("🔧 Using MoE_8e_2k_512d Model (baseline MoE - GLM4 MoE not available)")


class AttentionMoE_8e_2k_512dModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (8 experts, top-2, 512d) - matches target architecture"""
    
    def __init__(self, config: MoEModelConfig):
        # Override to 512 dimensions, 8 experts, top-2
        config.d_model = 512
        config.num_experts = 8
        config.expert_top_k = 2
        config.d_ff = 1024  # Smaller intermediate size for similar params
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_8e_2k_512d Model (DeepSeek Attention + GLM4 MoE: 8 experts, top-2, 512d)")


# =============================================================================
# REDUCED MODEL REGISTRY
# =============================================================================

REDUCED_ABLATION_MODELS = {
    # Baseline
    "baseline": BaselineModel,
    
    # MLP ablations (1 model) - only 512d
    "mlp_512d": MLP_512dModel,
    
    # Attention+MLP ablations (1 model) - only 512d
    "attention_mlp_512d": AttentionMLP_512dModel,
    
    # MoE ablations (1 model) - 8 experts, 512d
    "moe_8e_2k_512d": MoE_8e_2k_512dModel,
    
    # Attention+MoE ablations (1 model) - 8 experts, 512d
    "attention_moe_8e_2k_512d": AttentionMoE_8e_2k_512dModel,
}

# Total: 1 + 1 + 1 + 1 + 1 = 5 models


def create_reduced_ablation_model(model_name: str, config: MoEModelConfig):
    """Factory function to create reduced ablation models"""
    if model_name not in REDUCED_ABLATION_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(REDUCED_ABLATION_MODELS.keys())}")
    
    return REDUCED_ABLATION_MODELS[model_name](config)


def print_reduced_ablation_summary():
    """Print summary of all available reduced ablation models"""
    print(f"\n{'='*80}")
    print(f"🧪 REDUCED ABLATION STUDY - {len(REDUCED_ABLATION_MODELS)} MODELS")
    print(f"{'='*80}")
    
    categories = {
        "Baseline": ["baseline"],
        "MLP": ["mlp_512d"],
        "Attention+MLP": ["attention_mlp_512d"],
        "MoE": ["moe_8e_2k_512d"],
        "Attention+MoE": ["attention_moe_8e_2k_512d"]
    }
    
    for category, models in categories.items():
        print(f"\n📋 {category} ({len(models)} models):")
        for model in models:
            print(f"   • {model}")
    
    print(f"\n🎯 Total: {len(REDUCED_ABLATION_MODELS)} reduced ablation configurations")
    print(f"🔧 Focused on 512 hidden dimension scale")
    print(f"📏 Standard MLP scaling: 512d → 2048d inner")
    print(f"🧠 MoE configuration: 8 experts, top-2")
    print(f"📝 Target architecture: 512d, 8 experts, top-2")
    print(f"{'='*80}")


if __name__ == "__main__":
    print_reduced_ablation_summary()
