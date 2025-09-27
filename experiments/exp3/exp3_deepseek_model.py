"""
Experiment 3: DeepSeek Model with DeepseekV3RMSNorm
Minimal implementation using DeepSeek RMSNorm
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
from deepseek_modeling import DeepseekV3RMSNorm


class DeepSeekMoEModel(nn.Module):
    """DeepSeek MoE Model using DeepseekV3RMSNorm"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks with MoE
        self.transformer_blocks = nn.ModuleList([
            MoETransformerBlock(
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

        # Output layers using DeepSeek RMSNorm
        self.norm = DeepseekV3RMSNorm(config.d_model, eps=1e-6)
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
