# 06: Full Transformer in Code

The time has come. We've explored all the individual components in theory. Now, let's assemble them into a working Transformer model using PyTorch.

This code will bring together:
1.  The Token Embedding Layer
2.  Rotary Positional Embedding (RoPE)
3.  The Transformer Block, containing:
    *   Multi-Head Attention (with RoPE)
    *   A Mixture of Experts (MoE) Feed-Forward layer
    *   Pre-Normalization and Residual Connections
4.  The final Layer Normalization and Output Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

# -- 1. Model Configuration -- #

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None # For Grouped Query Attention, if needed
    vocab_size: int = 32000
    # MoE specific
    n_experts: int = 4
    n_experts_per_tok: int = 2
    # RoPE specific
    rope_theta: float = 10000.0

# -- 2. RoPE Implementation -- #

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, theta: float):
    # As per the RoPE paper, the frequencies are related to theta^(2k/d)
    # where k is the dimension index and d is the head dimension
    assert head_dim % 2 == 0, "head_dim must be even"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(seq_len)
    freqs = torch.outer(m, theta).float()
    # We can think of this as complex numbers polar(r, theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(1) # Add head dimension
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x)

# -- 3. MoE Implementation -- #

class Expert(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.dim * 2, bias=False)
        self.w2 = nn.Linear(args.dim * 2, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.dim * 2, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MOE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = nn.Linear(args.dim, args.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(args) for _ in range(args.n_experts)])
        self.n_experts_per_tok = args.n_experts_per_tok

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x = x.view(-1, dim)
        logits = self.gate(x)
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.n_experts_per_tok, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float).to(x.dtype)

        final_output = torch.zeros_like(x)
        flat_top_k_indices = top_k_indices.view(-1)
        
        # Combine expert outputs
        for i, expert in enumerate(self.experts):
            expert_mask = (flat_top_k_indices == i)
            if expert_mask.any():
                expert_input = x[expert_mask.nonzero(as_tuple=True)[0]]
                expert_output = expert(expert_input)
                # Weight the output
                weights = top_k_weights.view(-1)[expert_mask]
                final_output.index_add_(0, expert_mask.nonzero(as_tuple=True)[0], expert_output * weights.unsqueeze(1))

        return final_output.view(batch_size, seq_len, dim)

# -- 4. Attention & Transformer Block -- #

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_complex):
        batch_size, seq_len, _ = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE
        q = apply_rotary_embeddings(q, freqs_complex)
        k = apply_rotary_embeddings(k, freqs_complex)

        q = q.transpose(1, 2) # (bs, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        scores = scores.masked_fill(mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MOE(args)
        self.attention_norm = nn.LayerNorm(args.dim)
        self.ffn_norm = nn.LayerNorm(args.dim)

    def forward(self, x, freqs_complex):
        # Pre-normalization and residual connection for attention
        h = x + self.attention(self.attention_norm(x), freqs_complex)
        # Pre-normalization and residual connection for FFN
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# -- 5. The Full Transformer Model -- #

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, 2048, self.args.rope_theta) # Max seq len 2048

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Ensure freqs_complex is on the same device as the input
        self.freqs_complex = self.freqs_complex.to(h.device)
        freqs = self.freqs_complex[:seq_len]

        for layer in self.layers:
            h = layer(h, freqs)
        
        h = self.norm(h)
        # We only care about the output of the last token for generation
        output = self.output(h[:, -1, :])
        return output

# -- 6. Example Usage -- #

if __name__ == '__main__':
    args = ModelArgs(dim=256, n_layers=4, n_heads=4, vocab_size=1000)
    model = Transformer(args)

    # Tie weights (optional but good practice)
    model.tok_embeddings.weight = model.output.weight

    print(model)

    # Create a dummy input
    dummy_input = torch.randint(0, 1000, (2, 100)) # Batch size 2, sequence length 100
    
    # Get model output
    logits = model(dummy_input)
    print(f"Output logits shape: {logits.shape}") # Should be (batch_size, vocab_size)

```

### How to Read This Code

*   **`ModelArgs`**: We start by defining a configuration class. This makes it easy to change the model's hyperparameters (like size, number of layers, etc.) in one place.
*   **RoPE Functions**: `precompute_theta_pos_frequencies` does the math to prepare the rotation angles. `apply_rotary_embeddings` takes a query or key vector and applies the rotation.
*   **`MOE` and `Expert`**: These classes build our feed-forward layer, exactly as discussed in Module 8.
*   **`Attention`**: This module contains the core self-attention logic. It creates the Q, K, and V projections, applies RoPE to Q and K, performs the scaled dot-product attention with a causal mask, and projects the result back out.
*   **`TransformerBlock`**: This is the repeating unit of our model. It neatly packages one `Attention` layer and one `MOE` layer, connecting them with the Pre-Norm and residual connections we discussed.
*   **`Transformer`**: This is the top-level module. It initializes the token embeddings, creates a list of `TransformerBlock`s, and adds the final normalization and output layer. Its `forward` method defines the complete data flow from input token IDs to final logits.
*   **Example Usage**: The `if __name__ == '__main__'` block shows how to create an instance of the model and pass some dummy data through it to verify that the output shape is correct.

We now have a complete, functional Transformer model in a single file. The final step is to understand conceptually how we would train this model on a large dataset.
