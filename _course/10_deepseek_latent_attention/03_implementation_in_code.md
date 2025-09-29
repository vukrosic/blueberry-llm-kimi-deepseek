# 03: DeepSeek Latent Attention Implementation in Code

Now that we understand the theory and architecture behind DeepSeek's Latent Attention, let's translate that into a PyTorch implementation. We will create a module that encapsulates the three main attention steps: Input-to-Latent, Latent Self-Attention, and Latent-to-Input.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeepSeekLatentAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_latent_tokens: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_latent_tokens = n_latent_tokens

        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"

        # Learnable latent tokens
        self.latent_tokens = nn.Parameter(torch.randn(n_latent_tokens, dim))

        # Linear layers for Q, K, V projections
        # For Input-to-Latent and Latent-to-Input
        self.wq_input = nn.Linear(dim, dim, bias=False) # Query for input tokens
        self.wk_input = nn.Linear(dim, dim, bias=False) # Key for input tokens
        self.wv_input = nn.Linear(dim, dim, bias=False) # Value for input tokens

        self.wq_latent = nn.Linear(dim, dim, bias=False) # Query for latent tokens
        self.wk_latent = nn.Linear(dim, dim, bias=False) # Key for latent tokens
        self.wv_latent = nn.Linear(dim, dim, bias=False) # Value for latent tokens

        # Output projection
        self.wo = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Layer Normalization (optional, but good practice around attention)
        self.norm_input = nn.LayerNorm(dim)
        self.norm_latent = nn.LayerNorm(dim)

    def _attention(self, q, k, v, mask=None):
        # q, k, v are (batch_size, n_heads, seq_len, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        return output

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        # Apply LayerNorm to input tokens
        x_norm = self.norm_input(x)

        # Expand latent tokens to batch_size
        latent_tokens_batch = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        latent_tokens_norm = self.norm_latent(latent_tokens_batch)

        # --- 1. Input-to-Latent Attention (Compress input into latent tokens) ---
        # Queries from latent tokens, Keys/Values from input tokens
        q_latent_to_input = self.wq_latent(latent_tokens_norm).view(batch_size, self.n_latent_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k_input_for_latent = self.wk_input(x_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v_input_for_latent = self.wv_input(x_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # (batch_size, n_heads, n_latent_tokens, head_dim)
        latent_summary = self._attention(q_latent_to_input, k_input_for_latent, v_input_for_latent)
        latent_summary = latent_summary.transpose(1, 2).contiguous().view(batch_size, self.n_latent_tokens, self.dim)

        # --- 2. Latent Self-Attention (Refine latent tokens) ---
        # Queries, Keys, Values from latent_summary
        q_latent_self = self.wq_latent(latent_summary).view(batch_size, self.n_latent_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k_latent_self = self.wk_latent(latent_summary).view(batch_size, self.n_latent_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        v_latent_self = self.wv_latent(latent_summary).view(batch_size, self.n_latent_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        # (batch_size, n_heads, n_latent_tokens, head_dim)
        latent_refined = self._attention(q_latent_self, k_latent_self, v_latent_self)
        latent_refined = latent_refined.transpose(1, 2).contiguous().view(batch_size, self.n_latent_tokens, self.dim)

        # --- 3. Latent-to-Input Attention (Inject refined latent info back into input) ---
        # Queries from input tokens, Keys/Values from refined latent tokens
        q_input_from_latent = self.wq_input(x_norm).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k_latent_for_input = self.wk_latent(latent_refined).view(batch_size, self.n_latent_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        v_latent_for_input = self.wv_latent(latent_refined).view(batch_size, self.n_latent_tokens, self.n_heads, self.head_dim).transpose(1, 2)

        # (batch_size, n_heads, seq_len, head_dim)
        input_with_latent_context = self._attention(q_input_from_latent, k_latent_for_input, v_latent_for_input)
        input_with_latent_context = input_with_latent_context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        # Final output projection and residual connection
        output = self.wo(input_with_latent_context)
        return x + output # Residual connection

# Example Usage:
if __name__ == '__main__':
    dim = 512
    n_heads = 8
    n_latent_tokens = 64 # A small number of latent tokens
    seq_len = 1024 # A relatively long sequence
    batch_size = 2

    latent_attn = DeepSeekLatentAttention(dim, n_heads, n_latent_tokens)
    print(latent_attn)

    dummy_input = torch.randn(batch_size, seq_len, dim)
    print(f"Input shape: {dummy_input.shape}")

    output = latent_attn(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (batch_size, seq_len, dim)

    # Verify that the number of parameters is reasonable
    total_params = sum(p.numel() for p in latent_attn.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
```

This implementation provides a functional `DeepSeekLatentAttention` module that can be integrated into a Transformer block. It demonstrates how the learnable latent tokens mediate the information flow, allowing for efficient processing of long sequences by distilling global context.

This concludes our exploration of DeepSeek's Latent Attention. Next, we will move on to the GLM-4 Mixture of Experts module.
