import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import MixtureOfExperts


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create RoPE frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D]
        seq_len = x_BTHD.shape[-2]
        device = x_BTHD.device
        dtype = x_BTHD.dtype
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        
        # Apply RoPE - ensure proper broadcasting
        head_dim = x_BTHD.shape[-1]
        half_dim = head_dim // 2
        
        # Ensure cos and sin have the right shape for broadcasting
        cos = cos[..., :half_dim]
        sin = sin[..., :half_dim]
        
        x1, x2 = x_BTHD[..., :half_dim], x_BTHD[..., half_dim:]
        
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # B, T = x.size(0), x.size(1)
        # qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        # Q, K, V = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2] # [B, H, T, D]

        # Q = self.rotary(Q)
        # K = self.rotary(K)
        # Apply RoPE on [B, T, H, D]
        Q = self.rotary(Q.transpose(1, 2)).transpose(1, 2)
        K = self.rotary(K.transpose(1, 2)).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        # attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        return self.w_o(attn_output)


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Attention layer
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)

        # MoE layer
        self.feed_forward = MixtureOfExperts(
            d_model, d_ff, num_experts, top_k, dropout
        )

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss
