"""
Experiment 6: Comprehensive Exhaustive Ablation Study Models
Testing ALL possible combinations of DeepSeek components systematically
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
# COMPREHENSIVE ABLATION MODELS - ALL POSSIBLE COMBINATIONS
# =============================================================================

class BaselineAblationModel(MoEMinimalLLM):
    """Baseline model - no DeepSeek components (control group)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        print("🔧 Using Baseline Ablation Model (no DeepSeek components)")


# =============================================================================
# SINGLE COMPONENT ABLATIONS
# =============================================================================

class RMSNormAblationModel(MoEMinimalLLM):
    """Only DeepSeek RMSNorm"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace RMSNorm with DeepSeek RMSNorm
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
        
        print("🔧 Using RMSNorm Ablation Model (DeepSeek RMSNorm only)")


class MLPAblationModel(MoEMinimalLLM):
    """Only DeepSeek MLP"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace MLP with DeepSeek MLP
        deepseek_config = create_deepseek_config(config)
        for i, block in enumerate(self.transformer_blocks):
            block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using MLP Ablation Model (DeepSeek MLP only)")


class MoEAblationModel(MoEMinimalLLM):
    """Only GLM4 MoE"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace MoE with GLM4 MoE
        if GLM4_MOE_AVAILABLE:
            for i, block in enumerate(self.transformer_blocks):
                block.feed_forward = GLM4MoEWrapper(config)
            print("🔧 Using MoE Ablation Model (GLM4 MoE)")
        else:
            print("🔧 Using MoE Ablation Model (baseline MoE - GLM4 MoE not available)")


class AttentionAblationModel(MoEMinimalLLM):
    """Only Enhanced DeepSeek Attention"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace attention with DeepSeek attention (enhanced config)
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
        
        print("🔧 Using Attention Ablation Model (Enhanced DeepSeek Attention only)")


# =============================================================================
# TWO COMPONENT COMBINATIONS (6 total)
# =============================================================================

class RMSNormMLPAblationModel(MoEMinimalLLM):
    """DeepSeek RMSNorm + MLP"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both RMSNorm and MLP
        deepseek_config = create_deepseek_config(config)
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using RMSNorm+MLP Ablation Model (DeepSeek RMSNorm + MLP)")


class RMSNormMoEAblationModel(MoEMinimalLLM):
    """DeepSeek RMSNorm + GLM4 MoE"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both RMSNorm and MoE
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using RMSNorm+MoE Ablation Model (DeepSeek RMSNorm + GLM4 MoE)")


class RMSNormAttentionAblationModel(MoEMinimalLLM):
    """DeepSeek RMSNorm + Attention"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both RMSNorm and attention
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
        
        print("🔧 Using RMSNorm+Attention Ablation Model (DeepSeek RMSNorm + Attention)")


class MLPMoEAblationModel(MoEMinimalLLM):
    """DeepSeek MLP + GLM4 MoE (MoE overrides MLP)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both MLP and MoE (MoE overrides MLP)
        deepseek_config = create_deepseek_config(config)
        for i, block in enumerate(self.transformer_blocks):
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
            else:
                block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using MLP+MoE Ablation Model (GLM4 MoE - replaces MLP)")


class AttentionMLPAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + MLP"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both attention and MLP
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using Attention+MLP Ablation Model (DeepSeek Attention + MLP)")


class AttentionMoEAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both attention and MoE
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (DeepSeek Attention + GLM4 MoE)")


# =============================================================================
# THREE COMPONENT COMBINATIONS (4 total)
# =============================================================================

class RMSNormMLPMoEAblationModel(MoEMinimalLLM):
    """DeepSeek RMSNorm + MLP + GLM4 MoE (MoE overrides MLP)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace all three components (MoE overrides MLP)
        deepseek_config = create_deepseek_config(config)
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
            else:
                block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using RMSNorm+MLP+MoE Ablation Model (DeepSeek RMSNorm + GLM4 MoE)")


class RMSNormAttentionMLPAblationModel(MoEMinimalLLM):
    """DeepSeek RMSNorm + Attention + MLP"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace all three components
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using RMSNorm+Attention+MLP Ablation Model (DeepSeek RMSNorm + Attention + MLP)")


class RMSNormAttentionMoEAblationModel(MoEMinimalLLM):
    """DeepSeek RMSNorm + Attention + GLM4 MoE"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace all three components
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using RMSNorm+Attention+MoE Ablation Model (DeepSeek RMSNorm + Attention + GLM4 MoE)")


class AttentionMLPMoEAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + MLP + GLM4 MoE (MoE overrides MLP)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace all three components (MoE overrides MLP)
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
            else:
                block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using Attention+MLP+MoE Ablation Model (DeepSeek Attention + GLM4 MoE)")


# =============================================================================
# ALL FOUR COMPONENTS
# =============================================================================

class AllComponentsAblationModel(MoEMinimalLLM):
    """All DeepSeek components + GLM4 MoE - Best of all worlds"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace all components with DeepSeek versions + GLM4 MoE
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using All Components Ablation Model (DeepSeek components + GLM4 MoE)")


# =============================================================================
# ARCHITECTURE VARIATION ABLATIONS
# =============================================================================

class AttentionMoE_2LayersAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE with 2 layers"""
    
    def __init__(self, config: MoEModelConfig):
        # Override layer count
        config.n_layers = 2
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (2 layers)")


class AttentionMoE_4LayersAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE with 4 layers"""
    
    def __init__(self, config: MoEModelConfig):
        # Override layer count
        config.n_layers = 4
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (4 layers)")


class AttentionMoE_128dAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE with 128 dimensions"""
    
    def __init__(self, config: MoEModelConfig):
        # Override model dimension
        config.d_model = 128
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (128 dimensions)")


class AttentionMoE_512dAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE with 512 dimensions"""
    
    def __init__(self, config: MoEModelConfig):
        # Override model dimension
        config.d_model = 512
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (512 dimensions)")


class AttentionMoE_4ExpertsAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE with 4 experts"""
    
    def __init__(self, config: MoEModelConfig):
        # Override expert count
        config.num_experts = 4
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (4 experts)")


class AttentionMoE_16ExpertsAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE with 16 experts"""
    
    def __init__(self, config: MoEModelConfig):
        # Override expert count
        config.num_experts = 16
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (16 experts)")


class AttentionMoE_Top1AblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE with top-1 selection"""
    
    def __init__(self, config: MoEModelConfig):
        # Override top-k selection
        config.expert_top_k = 1
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (top-1 selection)")


class AttentionMoE_Top4AblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE with top-4 selection"""
    
    def __init__(self, config: MoEModelConfig):
        # Override top-k selection
        config.expert_top_k = 4
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (top-4 selection)")


# =============================================================================
# ATTENTION MECHANISM ABLATIONS
# =============================================================================

class AttentionNoRoPEAblationModel(MoEMinimalLLM):
    """DeepSeek Attention without RoPE scaling + GLM4 MoE"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = None  # No RoPE scaling
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (No RoPE scaling)")


class AttentionNoBiasAblationModel(MoEMinimalLLM):
    """DeepSeek Attention without bias + GLM4 MoE"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = False  # No attention bias
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (No attention bias)")


class AttentionLinearRoPEAblationModel(MoEMinimalLLM):
    """DeepSeek Attention with linear RoPE scaling + GLM4 MoE"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 2.0}  # 2x linear scaling
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE Ablation Model (2x Linear RoPE scaling)")


# =============================================================================
# COMPREHENSIVE MODEL REGISTRY
# =============================================================================

COMPREHENSIVE_ABLATION_MODELS = {
    # Baseline
    "baseline": BaselineAblationModel,
    
    # Single component ablations (4 models)
    "rmsnorm": RMSNormAblationModel,
    "mlp": MLPAblationModel,
    "moe": MoEAblationModel,
    "attention": AttentionAblationModel,
    
    # Two component combinations (6 models)
    "rmsnorm_mlp": RMSNormMLPAblationModel,
    "rmsnorm_moe": RMSNormMoEAblationModel,
    "rmsnorm_attention": RMSNormAttentionAblationModel,
    "mlp_moe": MLPMoEAblationModel,
    "attention_mlp": AttentionMLPAblationModel,
    "attention_moe": AttentionMoEAblationModel,
    
    # Three component combinations (4 models)
    "rmsnorm_mlp_moe": RMSNormMLPMoEAblationModel,
    "rmsnorm_attention_mlp": RMSNormAttentionMLPAblationModel,
    "rmsnorm_attention_moe": RMSNormAttentionMoEAblationModel,
    "attention_mlp_moe": AttentionMLPMoEAblationModel,
    
    # All four components
    "all_components": AllComponentsAblationModel,
    
    # Architecture variations (8 models)
    "attention_moe_2layers": AttentionMoE_2LayersAblationModel,
    "attention_moe_4layers": AttentionMoE_4LayersAblationModel,
    "attention_moe_128d": AttentionMoE_128dAblationModel,
    "attention_moe_512d": AttentionMoE_512dAblationModel,
    "attention_moe_4experts": AttentionMoE_4ExpertsAblationModel,
    "attention_moe_16experts": AttentionMoE_16ExpertsAblationModel,
    "attention_moe_top1": AttentionMoE_Top1AblationModel,
    "attention_moe_top4": AttentionMoE_Top4AblationModel,
    
    # Attention mechanism variations (3 models)
    "attention_moe_no_rope": AttentionNoRoPEAblationModel,
    "attention_moe_no_bias": AttentionNoBiasAblationModel,
    "attention_moe_linear_rope": AttentionLinearRoPEAblationModel,
}

# Total: 1 + 4 + 6 + 4 + 1 + 8 + 3 = 27 models


def create_comprehensive_ablation_model(model_name: str, config: MoEModelConfig):
    """Factory function to create comprehensive ablation models"""
    if model_name not in COMPREHENSIVE_ABLATION_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(COMPREHENSIVE_ABLATION_MODELS.keys())}")
    
    return COMPREHENSIVE_ABLATION_MODELS[model_name](config)


def print_comprehensive_ablation_summary():
    """Print summary of all available ablation models"""
    print(f"\n{'='*80}")
    print(f"🧪 COMPREHENSIVE ABLATION STUDY - {len(COMPREHENSIVE_ABLATION_MODELS)} MODELS")
    print(f"{'='*80}")
    
    categories = {
        "Baseline": ["baseline"],
        "Single Components": ["rmsnorm", "mlp", "moe", "attention"],
        "Two Components": ["rmsnorm_mlp", "rmsnorm_moe", "rmsnorm_attention", "mlp_moe", "attention_mlp", "attention_moe"],
        "Three Components": ["rmsnorm_mlp_moe", "rmsnorm_attention_mlp", "rmsnorm_attention_moe", "attention_mlp_moe"],
        "All Components": ["all_components"],
        "Architecture Variations": ["attention_moe_2layers", "attention_moe_4layers", "attention_moe_128d", "attention_moe_512d", "attention_moe_4experts", "attention_moe_16experts", "attention_moe_top1", "attention_moe_top4"],
        "Attention Variations": ["attention_moe_no_rope", "attention_moe_no_bias", "attention_moe_linear_rope"]
    }
    
    for category, models in categories.items():
        print(f"\n📋 {category} ({len(models)} models):")
        for model in models:
            print(f"   • {model}")
    
    print(f"\n🎯 Total: {len(COMPREHENSIVE_ABLATION_MODELS)} ablation configurations")
    print(f"{'='*80}")


if __name__ == "__main__":
    print_comprehensive_ablation_summary()
