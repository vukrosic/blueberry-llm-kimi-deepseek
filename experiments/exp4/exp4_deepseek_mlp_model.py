"""
Experiment 4: Model using DeepseekV3MLP
Minimal implementation using DeepseekV3MLP from deepseek_modeling.py
"""

import torch
import torch.nn as nn
import math
import sys
import os
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.moe_config import MoEModelConfig
from models.layers import MoETransformerBlock
from deepseek_modeling import DeepseekV3MLP
from configuration_deepseek import DeepseekV3Config


class DeepseekV3MLPMoEModel(nn.Module):
    """MoE Model using DeepseekV3MLP instead of standard MLP"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Create DeepseekV3Config for MLP
        deepseek_config = DeepseekV3Config(
            hidden_size=config.d_model,
            intermediate_size=config.d_ff,
            hidden_act="silu"  # Add this attribute for activation function
        )

        # Transformer blocks with MoE - we'll modify this to use DeepseekV3MLP
        self.transformer_blocks = nn.ModuleList([
            DeepseekV3MLPTransformerBlock(
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
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
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


class DeepseekV3MLPTransformerBlock(nn.Module):
    """Transformer block using DeepseekV3MLP instead of standard MLP"""
    
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, num_experts, expert_top_k, dropout, deepseek_config):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        self.dropout = dropout
        
        # Multi-head attention (keeping standard attention)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Use DeepseekV3MLP instead of standard MLP
        self.deepseek_mlp = DeepseekV3MLP(deepseek_config)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # MoE layer (keeping existing MoE structure)
        self.moe = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(num_experts)
        ])
        
        # Expert routing
        self.gate = nn.Linear(d_model, num_experts)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout_layer(attn_out))
        
        # MoE with DeepseekV3MLP
        # First apply DeepseekV3MLP
        deepseek_out = self.deepseek_mlp(x)
        
        # Then apply MoE routing
        gate_scores = self.gate(x)  # [batch, seq, num_experts]
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.expert_top_k, dim=-1)
        top_k_weights = torch.softmax(top_k_scores, dim=-1)
        
        # Apply MoE experts
        moe_out = torch.zeros_like(x)
        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i).any(dim=-1, keepdim=True)
            if expert_mask.any():
                expert_out = self.moe[i](x)
                expert_weight = torch.where(
                    top_k_indices == i,
                    top_k_weights,
                    torch.zeros_like(top_k_weights)
                ).sum(dim=-1, keepdim=True)
                moe_out += expert_out * expert_weight
        
        # Combine DeepseekV3MLP output with MoE output
        combined_out = (deepseek_out + moe_out) / 2.0
        
        # Residual connection and layer norm
        x = self.norm2(x + self.dropout_layer(combined_out))
        
        # Calculate auxiliary loss (load balancing)
        aux_loss = self._calculate_aux_loss(gate_scores)
        
        return x, aux_loss
    
    def _calculate_aux_loss(self, gate_scores):
        """Calculate auxiliary loss for load balancing"""
        if self.training:
            # Simple load balancing loss
            gate_probs = torch.softmax(gate_scores, dim=-1)
            expert_usage = gate_probs.mean(dim=(0, 1))  # Average usage per expert
            aux_loss = (expert_usage.std() ** 2) * 0.1  # Encourage balanced usage
            return aux_loss
        return None
