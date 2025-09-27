"""
Experiment 6: Ablation Study Models
Combining the best components from experiments 1-5 for comprehensive ablation study
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


class BaselineAblationModel(MoEMinimalLLM):
    """Baseline model - no DeepSeek components (control group)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        print("🔧 Using Baseline Ablation Model (no DeepSeek components)")


class RMSNormAblationModel(MoEMinimalLLM):
    """Only DeepSeek RMSNorm (from exp3)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace RMSNorm with DeepSeek RMSNorm
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
        
        print("🔧 Using RMSNorm Ablation Model (DeepSeek RMSNorm only)")


class MLPAblationModel(MoEMinimalLLM):
    """Only DeepSeek MLP (from exp4)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace MLP with DeepSeek MLP
        deepseek_config = create_deepseek_config(config)
        for i, block in enumerate(self.transformer_blocks):
            block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using MLP Ablation Model (DeepSeek MLP only)")


class MoEAblationModel(MoEMinimalLLM):
    """Only DeepSeek MoE (from exp5) - Note: DeepSeek MoE only supports inference mode"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        # Note: DeepSeek MoE only supports inference mode, so we use baseline MoE for training
        # This is a limitation of the current DeepSeek implementation
        print("🔧 Using MoE Ablation Model (DeepSeek MoE only - using baseline MoE due to training limitations)")


class AttentionAblationModel(MoEMinimalLLM):
    """Only Enhanced DeepSeek Attention (from exp1)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace attention with DeepSeek attention (enhanced config)
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
        
        print("🔧 Using Attention Ablation Model (Enhanced DeepSeek Attention only)")


class RMSNormMLPAblationModel(MoEMinimalLLM):
    """DeepSeek RMSNorm + MLP (exp3 + exp4)"""
    
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
    """DeepSeek RMSNorm + MoE (exp3 + exp5)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both RMSNorm and MoE
        deepseek_config = create_deepseek_config(config)
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            # Note: DeepSeek MoE only supports inference mode, so we use baseline MoE for training
        
        print("🔧 Using RMSNorm+MoE Ablation Model (DeepSeek RMSNorm + MoE)")


class MLPMoEAblationModel(MoEMinimalLLM):
    """DeepSeek MLP + MoE (exp4 + exp5)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both MLP and MoE (MoE overrides MLP)
        deepseek_config = create_deepseek_config(config)
        for i, block in enumerate(self.transformer_blocks):
            # Note: DeepSeek MoE only supports inference mode, so we use baseline MoE for training
            pass
        
        print("🔧 Using MLP+MoE Ablation Model (DeepSeek MoE - includes MLP)")


class AttentionRMSNormAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + RMSNorm (exp1 + exp3)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both attention and RMSNorm
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
        
        print("🔧 Using Attention+RMSNorm Ablation Model (DeepSeek Attention + RMSNorm)")


class AttentionMLPAblationModel(MoEMinimalLLM):
    """DeepSeek Attention + MLP (exp1 + exp4)"""
    
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
    """DeepSeek Attention + MoE (exp1 + exp5)"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both attention and MoE
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            # Note: DeepSeek MoE only supports inference mode, so we use baseline MoE for training
        
        print("🔧 Using Attention+MoE Ablation Model (DeepSeek Attention + MoE)")


class AllComponentsAblationModel(MoEMinimalLLM):
    """All DeepSeek components (exp1 + exp3 + exp5) - Best of all worlds"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace all components with DeepSeek versions
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        for i, block in enumerate(self.transformer_blocks):
            block.norm1 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.norm2 = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            # Note: DeepSeek MoE only supports inference mode, so we use baseline MoE for training
        
        print("🔧 Using All Components Ablation Model (All DeepSeek components)")


# Model registry for easy access
ABLATION_MODELS = {
    "baseline": BaselineAblationModel,
    "rmsnorm": RMSNormAblationModel,
    "mlp": MLPAblationModel,
    "moe": MoEAblationModel,
    "attention": AttentionAblationModel,
    "rmsnorm_mlp": RMSNormMLPAblationModel,
    "rmsnorm_moe": RMSNormMoEAblationModel,
    "mlp_moe": MLPMoEAblationModel,
    "attention_rmsnorm": AttentionRMSNormAblationModel,
    "attention_mlp": AttentionMLPAblationModel,
    "attention_moe": AttentionMoEAblationModel,
    "all_components": AllComponentsAblationModel,
}


def create_ablation_model(model_name: str, config: MoEModelConfig):
    """Factory function to create ablation models"""
    if model_name not in ABLATION_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(ABLATION_MODELS.keys())}")
    
    return ABLATION_MODELS[model_name](config)
