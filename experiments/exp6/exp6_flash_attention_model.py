"""
Experiment 6: DeepseekV3FlashAttention2 Model
Uses DeepseekV3FlashAttention2 from deepseek_modeling.py for comparison
"""

import torch
import torch.nn as nn
import math
import sys
import os
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.moe_config import MoEModelConfig
from deepseek_modeling import DeepseekV3Attention, DeepseekV3MLP
from configuration_deepseek import DeepseekV3Config
from models.components import MixtureOfExperts


class FlashAttentionModel(nn.Module):
    """Model using DeepseekV3FlashAttention2 for comparison with baseline attention"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        vocab_size = config.vocab_size if config.vocab_size is not None else 50257
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Create DeepseekV3Config for FlashAttention
        deepseek_config = DeepseekV3Config(
            hidden_size=config.d_model,
            intermediate_size=config.d_ff,
            hidden_act="silu",
            num_attention_heads=config.n_heads,
            max_position_embeddings=config.max_seq_len,
            attention_dropout=config.dropout,
            _attn_implementation="eager",  # Use standard DeepseekV3Attention
            # Disable MoE for this experiment (focus on attention)
            n_routed_experts=None,
            num_experts_per_tok=2,
        )

        # Create transformer blocks using DeepseekV3FlashAttention2
        self.transformer_blocks = nn.ModuleList([
            FlashAttentionTransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.max_seq_len,
                config.num_experts,
                config.expert_top_k,
                config.dropout,
                deepseek_config
            )
            for i in range(config.n_layers)
        ])

        # Output layers using standard RMSNorm (keeping baseline norm for fair comparison)
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_aux_loss=True):
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Collect auxiliary losses from MoE layers
        aux_losses = []

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x, aux_loss = block(x)
            if aux_loss is not None and return_aux_loss:
                aux_losses.append(aux_loss)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        # Combine auxiliary losses
        total_aux_loss = sum(aux_losses) if aux_losses else None

        if return_aux_loss:
            return logits, total_aux_loss
        return logits


class FlashAttentionTransformerBlock(nn.Module):
    """Transformer block using DeepseekV3FlashAttention2"""
    
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, num_experts, expert_top_k, dropout, deepseek_config):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Use DeepseekV3Attention instead of standard attention
        self.attention = DeepseekV3Attention(
            config=deepseek_config, 
            layer_idx=0  # We'll use same layer_idx for all blocks
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Use your existing MoE implementation (keeping MoE the same for fair comparison)
        self.moe = MixtureOfExperts(
            d_model, d_ff, num_experts, expert_top_k, dropout
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # DeepseekV3Attention forward pass
        # The FlashAttention expects specific input format, so we need to adapt
        attn_out, _, _ = self.attention(
            hidden_states=x,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False
        )
        
        # Residual connection and layer norm
        x = self.norm1(x + self.dropout_layer(attn_out))
        
        # MoE layer (keeping your existing implementation for fair comparison)
        moe_out, aux_loss = self.moe(x)
        
        # Residual connection and layer norm
        x = self.norm2(x + self.dropout_layer(moe_out))
        
        return x, aux_loss
