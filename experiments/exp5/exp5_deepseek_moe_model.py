"""
Experiment 5: DeepseekV3MoE Model
Uses DeepseekV3MoE from deepseek_modeling.py for comparison
"""

import torch
import torch.nn as nn
import math
import sys
import os
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.moe_config import MoEModelConfig
from deepseek_modeling import DeepseekV3MoE, DeepseekV3MLP
from configuration_deepseek import DeepseekV3Config


class DeepseekV3MoEModel(nn.Module):
    """Model using DeepseekV3MoE for comparison with baseline MoE"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        vocab_size = config.vocab_size if config.vocab_size is not None else 50257
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Create DeepseekV3Config for MoE
        deepseek_config = DeepseekV3Config(
            hidden_size=config.d_model,
            intermediate_size=config.d_ff,
            hidden_act="silu",
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.expert_top_k,
            moe_intermediate_size=config.d_ff,
            first_k_dense_replace=0,  # Use MoE from first layer
            moe_layer_freq=1,  # Use MoE in every layer
        )
        
        # Add MoE specific parameters that aren't in the base config
        deepseek_config.routed_scaling_factor = 1.0
        deepseek_config.scoring_func = "sigmoid"
        deepseek_config.seq_aux = False
        deepseek_config.topk_method = "topk"  # Use standard topk for training
        deepseek_config.n_group = 1
        deepseek_config.topk_group = 1
        deepseek_config.norm_topk_prob = True
        deepseek_config.n_shared_experts = None  # No shared experts for fair comparison

        # Create transformer blocks using DeepseekV3MoE
        self.transformer_blocks = nn.ModuleList([
            DeepseekV3MoETransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.max_seq_len,
                deepseek_config,
                config.dropout
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


class DeepseekV3MoETransformerBlock(nn.Module):
    """Transformer block using DeepseekV3MoE"""
    
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, deepseek_config, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        # Multi-head attention (keeping standard attention for fair comparison)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Use DeepseekV3MoE instead of standard MoE
        self.deepseek_moe = DeepseekV3MoE(deepseek_config)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout_layer(attn_out))
        
        # DeepseekV3MoE
        moe_out = self.deepseek_moe(x)
        
        # Residual connection and layer norm
        x = self.norm2(x + self.dropout_layer(moe_out))
        
        # For DeepseekV3MoE, we don't have explicit aux loss in this implementation
        # The load balancing is handled internally by the MoEGate
        aux_loss = None
        
        return x, aux_loss
