"""
Experiment 1: DeepSeek Attention Integration (Using Original Implementation)

This experiment imports and uses the original DeepSeek attention components
from deepseek_modeling.py to ensure correctness and minimize custom code.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

# Import original DeepSeek components
from deepseek_modeling import (
    DeepseekV3Attention,
    DeepseekV3FlashAttention2,
    DeepseekV3RMSNorm,
    ATTENTION_CLASSES
)
from configuration_deepseek import DeepseekV3Config
from models.components import MixtureOfExperts


class DeepSeekMoETransformerBlock(nn.Module):
    """Transformer block using original DeepSeek attention and our MoE"""
    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Use original DeepSeek attention
        self.attention = ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )
        
        # Use our MoE implementation
        self.feed_forward = MixtureOfExperts(
            config.hidden_size, 
            config.intermediate_size, 
            num_experts, 
            top_k, 
            dropout
        )
        
        # Use original DeepSeek normalization
        self.input_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, position_ids=None):
        # Self-attention (using original DeepSeek implementation)
        attn_out, _, _ = self.attention(
            hidden_states=self.input_layernorm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=False,
            use_cache=False
        )
        x = x + self.dropout(attn_out)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.post_attention_layernorm(x))
        x = x + self.dropout(ff_out)
        
        return x, aux_loss


class DeepSeekMoEModel(nn.Module):
    """MoE Model using original DeepSeek attention components"""
    def __init__(self, config: DeepseekV3Config, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_dropout = nn.Dropout(0.1)

        # Transformer blocks with original DeepSeek attention
        self.transformer_blocks = nn.ModuleList([
            DeepSeekMoETransformerBlock(
                config=config,
                layer_idx=i,
                num_experts=num_experts,
                top_k=top_k,
                dropout=0.1
            )
            for i in range(config.num_hidden_layers)
        ])

        # Output layers using original DeepSeek normalization
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(0.1)

        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_aux_loss=True, attention_mask=None, position_ids=None):
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.hidden_size)
        x = self.position_dropout(x)

        # Create attention mask if not provided
        if attention_mask is None:
            batch_size, seq_len = x.shape[:2]
            # Create 4D attention mask for DeepSeek attention
            attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=x.device, dtype=torch.float)
            # Fill with large negative values for causal mask
            attention_mask.masked_fill_(torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool(), float('-inf'))
        
        # Create position ids if not provided
        if position_ids is None:
            batch_size, seq_len = x.shape[:2]
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Collect auxiliary losses from MoE layers
        aux_losses = []

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x, aux_loss = block(x, attention_mask=attention_mask, position_ids=position_ids)
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
