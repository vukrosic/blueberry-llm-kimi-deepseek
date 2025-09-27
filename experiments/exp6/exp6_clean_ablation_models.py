"""
Experiment 6: Clean Ablation Study Models
Testing meaningful combinations without RMSNorm and with clear MoE naming
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
# CLEAN ABLATION MODELS - NO RMSNORM, CLEAR MOE NAMING
# =============================================================================

class BaselineAblationModel(MoEMinimalLLM):
    """Baseline model - no DeepSeek components (control group)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        print("🔧 Using Baseline Ablation Model (no DeepSeek components)")


# =============================================================================
# SINGLE COMPONENT ABLATIONS
# =============================================================================

class MLPAblationModel(MoEMinimalLLM):
    """Only DeepSeek MLP (replaces MoE with single MLP)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace MLP with DeepSeek MLP
        deepseek_config = create_deepseek_config(config)
        for i, block in enumerate(self.transformer_blocks):
            block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using MLP Ablation Model (DeepSeek MLP only - replaces MoE)")


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
        
        print("🔧 Using Attention Ablation Model (DeepSeek Attention only)")


# =============================================================================
# MOE ABLATIONS WITH CLEAR NAMING (experts_topk)
# =============================================================================

class MoE_8e_2kAblationModel(MoEMinimalLLM):
    """GLM4 MoE with 8 experts, top-2 selection"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Ensure 8 experts, top-2
        config.num_experts = 8
        config.expert_top_k = 2
        
        # Replace MoE with GLM4 MoE
        if GLM4_MOE_AVAILABLE:
            for i, block in enumerate(self.transformer_blocks):
                block.feed_forward = GLM4MoEWrapper(config)
            print("🔧 Using MoE_8e_2k Ablation Model (GLM4 MoE: 8 experts, top-2)")
        else:
            print("🔧 Using MoE_8e_2k Ablation Model (baseline MoE - GLM4 MoE not available)")


class MoE_4e_2kAblationModel(MoEMinimalLLM):
    """GLM4 MoE with 4 experts, top-2 selection"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Override to 4 experts, top-2
        config.num_experts = 4
        config.expert_top_k = 2
        
        # Replace MoE with GLM4 MoE
        if GLM4_MOE_AVAILABLE:
            for i, block in enumerate(self.transformer_blocks):
                block.feed_forward = GLM4MoEWrapper(config)
            print("🔧 Using MoE_4e_2k Ablation Model (GLM4 MoE: 4 experts, top-2)")
        else:
            print("🔧 Using MoE_4e_2k Ablation Model (baseline MoE - GLM4 MoE not available)")


class MoE_8e_1kAblationModel(MoEMinimalLLM):
    """GLM4 MoE with 8 experts, top-1 selection"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Override to 8 experts, top-1
        config.num_experts = 8
        config.expert_top_k = 1
        
        # Replace MoE with GLM4 MoE
        if GLM4_MOE_AVAILABLE:
            for i, block in enumerate(self.transformer_blocks):
                block.feed_forward = GLM4MoEWrapper(config)
            print("🔧 Using MoE_8e_1k Ablation Model (GLM4 MoE: 8 experts, top-1)")
        else:
            print("🔧 Using MoE_8e_1k Ablation Model (baseline MoE - GLM4 MoE not available)")


class MoE_16e_2kAblationModel(MoEMinimalLLM):
    """GLM4 MoE with 16 experts, top-2 selection"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Override to 16 experts, top-2
        config.num_experts = 16
        config.expert_top_k = 2
        
        # Replace MoE with GLM4 MoE
        if GLM4_MOE_AVAILABLE:
            for i, block in enumerate(self.transformer_blocks):
                block.feed_forward = GLM4MoEWrapper(config)
            print("🔧 Using MoE_16e_2k Ablation Model (GLM4 MoE: 16 experts, top-2)")
        else:
            print("🔧 Using MoE_16e_2k Ablation Model (baseline MoE - GLM4 MoE not available)")


# =============================================================================
# TWO COMPONENT COMBINATIONS
# =============================================================================

class AttentionMLPAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + MLP (MLP replaces MoE)"""
    
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


class AttentionMoE_8e_2kAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (8 experts, top-2)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Ensure 8 experts, top-2
        config.num_experts = 8
        config.expert_top_k = 2
        
        # Replace both attention and MoE
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_8e_2k Ablation Model (DeepSeek Attention + GLM4 MoE: 8 experts, top-2)")


class AttentionMoE_4e_2kAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (4 experts, top-2)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Override to 4 experts, top-2
        config.num_experts = 4
        config.expert_top_k = 2
        
        # Replace both attention and MoE
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_4e_2k Ablation Model (DeepSeek Attention + GLM4 MoE: 4 experts, top-2)")


class AttentionMoE_8e_1kAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (8 experts, top-1)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Override to 8 experts, top-1
        config.num_experts = 8
        config.expert_top_k = 1
        
        # Replace both attention and MoE
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_8e_1k Ablation Model (DeepSeek Attention + GLM4 MoE: 8 experts, top-1)")


class AttentionMoE_16e_2kAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (16 experts, top-2)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Override to 16 experts, top-2
        config.num_experts = 16
        config.expert_top_k = 2
        
        # Replace both attention and MoE
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_16e_2k Ablation Model (DeepSeek Attention + GLM4 MoE: 16 experts, top-2)")


# =============================================================================
# ARCHITECTURE SCALING ABLATIONS
# =============================================================================

class AttentionMoE_8e_2k_256dAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (8 experts, top-2, 256d)"""
    
    def __init__(self, config: MoEModelConfig):
        # Ensure 256 dimensions, 8 experts, top-2
        config.d_model = 256
        config.num_experts = 8
        config.expert_top_k = 2
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_8e_2k_256d Ablation Model (DeepSeek Attention + GLM4 MoE: 8 experts, top-2, 256d)")


class AttentionMoE_8e_2k_512dAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (8 experts, top-2, 512d)"""
    
    def __init__(self, config: MoEModelConfig):
        # Override to 512 dimensions, 8 experts, top-2
        config.d_model = 512
        config.num_experts = 8
        config.expert_top_k = 2
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_8e_2k_512d Ablation Model (DeepSeek Attention + GLM4 MoE: 8 experts, top-2, 512d)")


class AttentionMoE_8e_2k_1024dAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (8 experts, top-2, 1024d)"""
    
    def __init__(self, config: MoEModelConfig):
        # Override to 1024 dimensions, 8 experts, top-2
        config.d_model = 1024
        config.num_experts = 8
        config.expert_top_k = 2
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_8e_2k_1024d Ablation Model (DeepSeek Attention + GLM4 MoE: 8 experts, top-2, 1024d)")


class AttentionMoE_4e_2k_512dAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (4 experts, top-2, 512d)"""
    
    def __init__(self, config: MoEModelConfig):
        # Override to 512 dimensions, 4 experts, top-2
        config.d_model = 512
        config.num_experts = 4
        config.expert_top_k = 2
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_4e_2k_512d Ablation Model (DeepSeek Attention + GLM4 MoE: 4 experts, top-2, 512d)")


class AttentionMoE_16e_2k_512dAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (16 experts, top-2, 512d)"""
    
    def __init__(self, config: MoEModelConfig):
        # Override to 512 dimensions, 16 experts, top-2
        config.d_model = 512
        config.num_experts = 16
        config.expert_top_k = 2
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_16e_2k_512d Ablation Model (DeepSeek Attention + GLM4 MoE: 16 experts, top-2, 512d)")


# =============================================================================
# LAYER COUNT ABLATIONS
# =============================================================================

class AttentionMoE_8e_2k_3layersAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (8 experts, top-2, 3 layers)"""
    
    def __init__(self, config: MoEModelConfig):
        # Ensure 3 layers, 8 experts, top-2
        config.n_layers = 3
        config.num_experts = 8
        config.expert_top_k = 2
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_8e_2k_3layers Ablation Model (DeepSeek Attention + GLM4 MoE: 8 experts, top-2, 3 layers)")


class AttentionMoE_8e_2k_6layersAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + GLM4 MoE (8 experts, top-2, 6 layers)"""
    
    def __init__(self, config: MoEModelConfig):
        # Override to 6 layers, 8 experts, top-2
        config.n_layers = 6
        config.num_experts = 8
        config.expert_top_k = 2
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_8e_2k_6layers Ablation Model (DeepSeek Attention + GLM4 MoE: 8 experts, top-2, 6 layers)")


# =============================================================================
# ATTENTION MECHANISM ABLATIONS
# =============================================================================

class AttentionMoE_8e_2k_NoRoPEAblationModel(MoEMinimalLLM):
    """DeepSeek Attention (no RoPE) + GLM4 MoE (8 experts, top-2)"""
    
    def __init__(self, config: MoEModelConfig):
        # Ensure 8 experts, top-2
        config.num_experts = 8
        config.expert_top_k = 2
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = None  # No RoPE scaling
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_8e_2k_NoRoPE Ablation Model (DeepSeek Attention no RoPE + GLM4 MoE: 8 experts, top-2)")


class AttentionMoE_8e_2k_NoBiasAblationModel(MoEMinimalLLM):
    """DeepSeek Attention (no bias) + GLM4 MoE (8 experts, top-2)"""
    
    def __init__(self, config: MoEModelConfig):
        # Ensure 8 experts, top-2
        config.num_experts = 8
        config.expert_top_k = 2
        super().__init__(config)
        
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = False  # No attention bias
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Attention+MoE_8e_2k_NoBias Ablation Model (DeepSeek Attention no bias + GLM4 MoE: 8 experts, top-2)")


class AttentionMoE_8e_2k_StandardAblationModel(MoEMinimalLLM):
    """Standard Attention + GLM4 MoE (8 experts, top-2) - no DeepSeek attention features"""
    
    def __init__(self, config: MoEModelConfig):
        # Ensure 8 experts, top-2
        config.num_experts = 8
        config.expert_top_k = 2
        super().__init__(config)
        
        # Keep standard attention, only replace MoE
        for i, block in enumerate(self.transformer_blocks):
            if GLM4_MOE_AVAILABLE:
                block.feed_forward = GLM4MoEWrapper(config)
        
        print("🔧 Using Standard+MoE_8e_2k Ablation Model (Standard Attention + GLM4 MoE: 8 experts, top-2)")


# =============================================================================
# CLEAN MODEL REGISTRY
# =============================================================================

CLEAN_ABLATION_MODELS = {
    # Baseline
    "baseline": BaselineAblationModel,
    
    # Single component ablations (2 models)
    "mlp": MLPAblationModel,
    "attention": AttentionAblationModel,
    
    # MoE ablations with clear naming (4 models)
    "moe_8e_2k": MoE_8e_2kAblationModel,
    "moe_4e_2k": MoE_4e_2kAblationModel,
    "moe_8e_1k": MoE_8e_1kAblationModel,
    "moe_16e_2k": MoE_16e_2kAblationModel,
    
    # Two component combinations (5 models)
    "attention_mlp": AttentionMLPAblationModel,
    "attention_moe_8e_2k": AttentionMoE_8e_2kAblationModel,
    "attention_moe_4e_2k": AttentionMoE_4e_2kAblationModel,
    "attention_moe_8e_1k": AttentionMoE_8e_1kAblationModel,
    "attention_moe_16e_2k": AttentionMoE_16e_2kAblationModel,
    
    # Architecture scaling (5 models)
    "attention_moe_8e_2k_256d": AttentionMoE_8e_2k_256dAblationModel,
    "attention_moe_8e_2k_512d": AttentionMoE_8e_2k_512dAblationModel,
    "attention_moe_8e_2k_1024d": AttentionMoE_8e_2k_1024dAblationModel,
    "attention_moe_4e_2k_512d": AttentionMoE_4e_2k_512dAblationModel,
    "attention_moe_16e_2k_512d": AttentionMoE_16e_2k_512dAblationModel,
    
    # Layer count ablations (2 models)
    "attention_moe_8e_2k_3layers": AttentionMoE_8e_2k_3layersAblationModel,
    "attention_moe_8e_2k_6layers": AttentionMoE_8e_2k_6layersAblationModel,
    
    # Attention mechanism ablations (3 models)
    "attention_moe_8e_2k_no_rope": AttentionMoE_8e_2k_NoRoPEAblationModel,
    "attention_moe_8e_2k_no_bias": AttentionMoE_8e_2k_NoBiasAblationModel,
    "standard_moe_8e_2k": AttentionMoE_8e_2k_StandardAblationModel,
}

# Total: 1 + 2 + 4 + 5 + 5 + 2 + 3 = 22 models


def create_clean_ablation_model(model_name: str, config: MoEModelConfig):
    """Factory function to create clean ablation models"""
    if model_name not in CLEAN_ABLATION_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(CLEAN_ABLATION_MODELS.keys())}")
    
    return CLEAN_ABLATION_MODELS[model_name](config)


def print_clean_ablation_summary():
    """Print summary of all available clean ablation models"""
    print(f"\n{'='*80}")
    print(f"🧪 CLEAN ABLATION STUDY - {len(CLEAN_ABLATION_MODELS)} MODELS")
    print(f"{'='*80}")
    
    categories = {
        "Baseline": ["baseline"],
        "Single Components": ["mlp", "attention"],
        "MoE Configurations": ["moe_8e_2k", "moe_4e_2k", "moe_8e_1k", "moe_16e_2k"],
        "Two Components": ["attention_mlp", "attention_moe_8e_2k", "attention_moe_4e_2k", "attention_moe_8e_1k", "attention_moe_16e_2k"],
        "Architecture Scaling": ["attention_moe_8e_2k_256d", "attention_moe_8e_2k_512d", "attention_moe_8e_2k_1024d", "attention_moe_4e_2k_512d", "attention_moe_16e_2k_512d"],
        "Layer Count": ["attention_moe_8e_2k_3layers", "attention_moe_8e_2k_6layers"],
        "Attention Variants": ["attention_moe_8e_2k_no_rope", "attention_moe_8e_2k_no_bias", "standard_moe_8e_2k"]
    }
    
    for category, models in categories.items():
        print(f"\n📋 {category} ({len(models)} models):")
        for model in models:
            print(f"   • {model}")
    
    print(f"\n🎯 Total: {len(CLEAN_ABLATION_MODELS)} clean ablation configurations")
    print(f"🔧 No RMSNorm ablations (not useful)")
    print(f"📝 Clear MoE naming: moe_{'{experts}e_{topk}k'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    print_clean_ablation_summary()
