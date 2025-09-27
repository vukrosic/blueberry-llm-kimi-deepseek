"""
Experiment 7: Best Architecture Model
Uses the attention_mlp architecture that achieved the best efficiency in exp6
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.moe_llm import MoEMinimalLLM
from deepseek_modeling import DeepseekV3Attention, DeepseekV3MLP, DeepseekV3RMSNorm
from configuration_deepseek import DeepseekV3Config
from configs.moe_config import MoEModelConfig


class DeepseekV3MLPWrapper(nn.Module):
    """Wrapper for DeepseekV3MLP to return (output, aux_loss) format"""
    
    def __init__(self, config):
        super().__init__()
        self.mlp = DeepseekV3MLP(config)
    
    def forward(self, x):
        output = self.mlp(x)
        return output, None  # Return (output, aux_loss) format expected by MoETransformerBlock


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
        # MoE specific (not used in MLP model)
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
        # Additional required attributes
        seq_aux=True,
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
    )


class Exp7AttentionMLPModel(MoEMinimalLLM):
    """
    Experiment 7 Model: Best Architecture from exp6
    Uses DeepSeek Attention + DeepSeek MLP (attention_mlp architecture)
    This achieved the best efficiency in exp6: 0.0364 loss, 0.65 min training, 15.59M params
    """
    
    def __init__(self, config: MoEModelConfig):
        super().__init__(config)
        
        # Replace both attention and MLP with DeepSeek versions
        deepseek_config = create_deepseek_config(config)
        deepseek_config.rope_scaling = {"type": "linear", "factor": 1.0}
        deepseek_config.attention_bias = True
        
        for i, block in enumerate(self.transformer_blocks):
            # Replace attention with DeepSeek attention
            block.attention = DeepseekV3AttentionWrapper(deepseek_config, layer_idx=i)
            # Replace MoE feedforward with DeepSeek MLP
            block.feed_forward = DeepseekV3MLPWrapper(deepseek_config)
        
        print("🔧 Using Exp7 Attention+MLP Model (Best Architecture from exp6)")
        print(f"   - DeepSeek Attention with LoRA + enhanced RoPE + attention bias")
        print(f"   - DeepSeek MLP with SiLU + gated architecture")
        print(f"   - Expected: 15.59M params, ~0.65 min training, ~0.0364 loss")


def create_exp7_model(config: MoEModelConfig) -> Exp7AttentionMLPModel:
    """Create the exp7 model with best architecture from exp6"""
    return Exp7AttentionMLPModel(config)
