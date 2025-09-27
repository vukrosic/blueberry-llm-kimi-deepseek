"""
Experiment 5: Simplified DeepseekV3MoE Model
Uses DeepseekV3MLP experts but with simpler gating for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.moe_config import MoEModelConfig
from deepseek_modeling import DeepseekV3MLP
from configuration_deepseek import DeepseekV3Config


class SimpleDeepseekV3MoE(nn.Module):
    """Simplified MoE using DeepseekV3MLP experts with standard gating"""
    
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.expert_top_k
        
        # Create DeepseekV3Config for MLP experts
        deepseek_config = DeepseekV3Config(
            hidden_size=config.d_model,
            intermediate_size=config.d_ff,
            hidden_act="silu"
        )
        
        # Create experts using DeepseekV3MLP
        self.experts = nn.ModuleList([
            DeepseekV3MLP(deepseek_config) for _ in range(self.num_experts)
        ])
        
        # Simple gating network
        self.gate = nn.Linear(config.d_model, self.num_experts, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Compute gate scores
        gate_scores = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens that use this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get tokens for this expert
            expert_tokens = x[expert_mask]  # [num_tokens, d_model]
            
            # Apply expert
            expert_out = self.experts[expert_idx](expert_tokens)
            
            # Get weights for this expert
            expert_weights = torch.where(
                top_k_indices == expert_idx,
                top_k_weights,
                torch.zeros_like(top_k_weights)
            ).sum(dim=-1, keepdim=True)
            
            # Weight the output
            weighted_out = expert_out * expert_weights[expert_mask]
            output[expert_mask] += weighted_out
        
        return output


class SimpleDeepseekV3MoEModel(nn.Module):
    """Model using simplified DeepseekV3MoE for comparison with baseline MoE"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        vocab_size = config.vocab_size if config.vocab_size is not None else 50257
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Create transformer blocks using simplified DeepseekV3MoE
        self.transformer_blocks = nn.ModuleList([
            SimpleDeepseekV3MoETransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.max_seq_len,
                config.num_experts,
                config.expert_top_k,
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


class SimpleDeepseekV3MoETransformerBlock(nn.Module):
    """Transformer block using simplified DeepseekV3MoE"""
    
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, num_experts, expert_top_k, dropout):
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
        
        # Use simplified DeepseekV3MoE
        self.deepseek_moe = SimpleDeepseekV3MoE(
            type('Config', (), {
                'd_model': d_model,
                'd_ff': d_ff,
                'num_experts': num_experts,
                'expert_top_k': expert_top_k
            })()
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout_layer(attn_out))
        
        # Simplified DeepseekV3MoE
        moe_out = self.deepseek_moe(x)
        
        # Residual connection and layer norm
        x = self.norm2(x + self.dropout_layer(moe_out))
        
        # Calculate simple auxiliary loss for load balancing
        aux_loss = self._calculate_aux_loss(x)
        
        return x, aux_loss
    
    def _calculate_aux_loss(self, x):
        """Calculate simple load balancing loss"""
        if self.training:
            # Get gate scores from the MoE layer
            gate_scores = self.deepseek_moe.gate(x)
            gate_probs = F.softmax(gate_scores, dim=-1)
            expert_usage = gate_probs.mean(dim=(0, 1))
            aux_loss = (expert_usage.std() ** 2) * 0.01
            return aux_loss
        return None
