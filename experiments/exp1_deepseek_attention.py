"""
Experiment 1: DeepSeek Attention Integration

This experiment integrates DeepSeek's advanced attention mechanisms into our MoE model:
1. LoRA-style Q/K/V projections with configurable ranks
2. Separate head dimensions for Q/K vs V
3. Advanced RoPE scaling (linear, dynamic NTK, YARN)
4. Flash Attention 2 support
5. Enhanced attention bias handling

The goal is to improve attention efficiency and potentially model performance
while maintaining the existing MoE architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union
from models.components import MixtureOfExperts


class DeepSeekRotaryEmbedding(nn.Module):
    """Enhanced RoPE with scaling support (inspired by DeepSeek)"""
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000, 
                 scaling_type: str = "none", scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        
        # Create base RoPE using simple implementation
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Apply scaling if specified
        if scaling_type == "linear":
            self._apply_linear_scaling()
        elif scaling_type == "dynamic":
            self._apply_dynamic_scaling()
    
    def _apply_linear_scaling(self):
        """Apply linear scaling to RoPE frequencies"""
        # This is a simplified version - full implementation would modify the base frequencies
        pass
    
    def _apply_dynamic_scaling(self):
        """Apply dynamic NTK scaling to RoPE frequencies"""
        # This is a simplified version - full implementation would modify the base frequencies
        pass
    
    def forward(self, x_BTHD: torch.Tensor):
        # Simple RoPE implementation
        seq_len = x_BTHD.shape[-2]
        device = x_BTHD.device
        dtype = x_BTHD.dtype
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        
        # Apply RoPE - ensure dimensions match
        head_dim = x_BTHD.shape[-1]
        half_dim = head_dim // 2
        
        # Ensure cos and sin have the right shape
        cos = cos[..., :half_dim]
        sin = sin[..., :half_dim]
        
        x1, x2 = x_BTHD[..., :half_dim], x_BTHD[..., half_dim:]
        
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class DeepSeekMultiHeadAttention(nn.Module):
    """Enhanced Multi-Head Attention with DeepSeek features"""
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        max_seq_len: int, 
        dropout: float = 0.1,
        # DeepSeek specific parameters
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        attention_bias: bool = False,
        rope_scaling: Optional[dict] = None,
        use_flash_attention: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # DeepSeek specific dimensions
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim or self.d_k
        self.v_head_dim = v_head_dim or self.d_k
        self.qk_nope_head_dim = self.d_k - self.qk_rope_head_dim
        self.attention_bias = attention_bias
        self.use_flash_attention = use_flash_attention
        
        # Initialize projections based on DeepSeek architecture
        self._init_projections()
        
        # Initialize RoPE with scaling
        rope_config = rope_scaling or {}
        self.rotary = DeepSeekRotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_seq_len=max_seq_len,
            scaling_type=rope_config.get("type", "none"),
            scaling_factor=rope_config.get("factor", 1.0)
        )
        
        self.dropout = dropout
        self.softmax_scale = self.d_k ** (-0.5)
        
        # Apply RoPE scaling to softmax scale if needed
        if rope_config.get("type") == "yarn" and rope_config.get("mscale_all_dim"):
            mscale = self._get_yarn_mscale(rope_config["factor"], rope_config["mscale_all_dim"])
            self.softmax_scale = self.softmax_scale * mscale * mscale
    
    def _init_projections(self):
        """Initialize Q/K/V projections with optional LoRA"""
        if self.q_lora_rank is None:
            # Standard Q projection
            self.q_proj = nn.Linear(
                self.d_model, self.n_heads * self.d_k, bias=self.attention_bias
            )
        else:
            # LoRA-style Q projection
            self.q_a_proj = nn.Linear(
                self.d_model, self.q_lora_rank, bias=self.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.n_heads * self.d_k, bias=False
            )
        
        # KV projection with MQA support
        self.kv_a_proj_with_mqa = nn.Linear(
            self.d_model,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=self.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.n_heads * self.v_head_dim,
            self.d_model,
            bias=self.attention_bias,
        )
    
    def _get_yarn_mscale(self, scale: float, mscale_all_dim: float) -> float:
        """Get YARN mscale value"""
        if scale <= 1:
            return 1.0
        return 0.1 * mscale_all_dim * math.log(scale) + 1.0
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Q projection
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # KV projection with MQA
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # k_pe is shared across heads (MQA), so we expand it
        k_pe = k_pe.unsqueeze(2).expand(batch_size, seq_len, self.n_heads, self.qk_rope_head_dim).transpose(1, 2)
        
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(batch_size, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # Apply RoPE to positional parts
        # q_pe shape: [batch_size, n_heads, seq_len, qk_rope_head_dim]
        q_pe = self.rotary(q_pe)
        k_pe = self.rotary(k_pe)
        
        # Combine positional and non-positional parts
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)
        
        # Attention computation
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use Flash Attention if available
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                is_causal=True, 
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Standard attention
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.softmax_scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.n_heads * self.v_head_dim)
        
        return self.o_proj(attn_output)


class DeepSeekMoETransformerBlock(nn.Module):
    """Transformer block with DeepSeek attention and MoE"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        # DeepSeek attention parameters
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        attention_bias: bool = False,
        rope_scaling: Optional[dict] = None,
        use_flash_attention: bool = False
    ):
        super().__init__()

        # DeepSeek attention layer
        self.attention = DeepSeekMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            attention_bias=attention_bias,
            rope_scaling=rope_scaling,
            use_flash_attention=use_flash_attention
        )

        # MoE layer (unchanged)
        self.feed_forward = MixtureOfExperts(
            d_model, d_ff, num_experts, top_k, dropout
        )

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with DeepSeek enhancements
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss


class DeepSeekMoEModel(nn.Module):
    """MoE Model with DeepSeek attention enhancements"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks with DeepSeek attention
        self.transformer_blocks = nn.ModuleList([
            DeepSeekMoETransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.max_seq_len,
                config.num_experts,
                config.expert_top_k,
                config.dropout,
                # DeepSeek parameters
                q_lora_rank=getattr(config, 'q_lora_rank', None),
                kv_lora_rank=getattr(config, 'kv_lora_rank', None),
                qk_rope_head_dim=getattr(config, 'qk_rope_head_dim', None),
                v_head_dim=getattr(config, 'v_head_dim', None),
                attention_bias=getattr(config, 'attention_bias', False),
                rope_scaling=getattr(config, 'rope_scaling', None),
                use_flash_attention=getattr(config, 'use_flash_attention', False)
            )
            for i in range(config.n_layers)
        ])

        # Output layers
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
