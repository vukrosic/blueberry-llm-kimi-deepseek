# 03: GLM-4 MoE Implementation in Code

We've covered the theoretical and architectural aspects of GLM-4's Mixture of Experts. Now, let's put it into practice by implementing a GLM-4 style MoE layer in PyTorch. This implementation will include the router, the experts, and the logic for routing tokens to the top-K experts.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A simple Feed-Forward Network (FFN) that acts as an expert."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.GELU() # Common activation in modern LLMs
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))

class GLM4MoE(nn.Module):
    """GLM-4 style Mixture of Experts layer."""
    def __init__(self, dim: int, hidden_dim: int, n_experts: int, top_k: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.top_k = top_k

        assert self.top_k <= self.n_experts, "top_k cannot be greater than n_experts"

        # Router (Gating Network)
        self.gate = nn.Linear(dim, n_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        original_shape = x.shape
        x = x.view(-1, self.dim) # Flatten for easier routing: (num_tokens, dim)
        num_tokens = x.shape[0]

        # Get expert logits from the gate
        gate_logits = self.gate(x) # (num_tokens, n_experts)

        # Select top-k experts
        # top_k_logits: (num_tokens, top_k) - scores of selected experts
        # top_k_indices: (num_tokens, top_k) - indices of selected experts
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)

        # Convert logits to probabilities using softmax
        # These probabilities will be used to weight the expert outputs
        top_k_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float).to(x.dtype)

        # Initialize output tensor
        final_output = torch.zeros_like(x) # (num_tokens, dim)

        # Create a mask to track which tokens go to which expert
        # This is useful for load balancing (not implemented here, but conceptually important)
        # expert_counts = torch.zeros(self.n_experts, dtype=torch.long, device=x.device)
        # token_expert_assignment = torch.zeros(num_tokens, self.n_experts, dtype=torch.bool, device=x.device)

        # Iterate through each expert
        for i, expert in enumerate(self.experts):
            # Find all tokens that are routed to this expert 'i'
            # expert_mask: (num_tokens, top_k) boolean tensor
            expert_mask = (top_k_indices == i)
            
            # Get the indices of tokens that are routed to expert 'i'
            # flat_expert_mask_indices: (num_tokens_for_this_expert,) tensor
            flat_expert_mask_indices = expert_mask.nonzero(as_tuple=True)[0]

            if flat_expert_mask_indices.shape[0] > 0:
                # Get the input tokens for this expert
                expert_input = x[flat_expert_mask_indices]
                
                # Process with the expert
                expert_output = expert(expert_input)
                
                # Get the corresponding weights for these tokens and this expert
                # We need to find the position of expert 'i' within the top_k_indices for each token
                # This is a bit tricky: we need to get the weight from top_k_weights
                # where top_k_indices == i
                
                # Create a tensor of weights for the current expert 'i'
                weights_for_this_expert = top_k_weights[expert_mask]

                # Add the weighted expert output to the final output
                final_output.index_add_(0, flat_expert_mask_indices, expert_output * weights_for_this_expert.unsqueeze(1))

        return final_output.view(original_shape)

# Example Usage:
if __name__ == '__main__':
    dim = 512
    hidden_dim = 2048 # Typically 4 * dim or similar
    n_experts = 8
    top_k = 2
    seq_len = 100
    batch_size = 4

    moe_layer = GLM4MoE(dim, hidden_dim, n_experts, top_k)
    print(moe_layer)

    dummy_input = torch.randn(batch_size, seq_len, dim)
    print(f"Input shape: {dummy_input.shape}")

    output = moe_layer(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (batch_size, seq_len, dim)

    # Verify that the number of parameters is reasonable
    total_params = sum(p.numel() for p in moe_layer.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Compare with a dense FFN of similar capacity (e.g., if all experts were active)
    # A dense FFN would have (dim * hidden_dim + hidden_dim * dim) parameters
    # For one expert: (512*2048 + 2048*512) = 2,097,152
    # For MoE with 8 experts, top_k=2, the active parameters are roughly 2 * (expert_params) + gate_params
    # Gate params: dim * n_experts = 512 * 8 = 4096
    # Total active params: 2 * 2,097,152 + 4096 = ~4.2M (for top-2)
    # Total potential params: 8 * 2,097,152 + 4096 = ~16.7M
    # This demonstrates the sparse activation benefit.
```

This implementation provides a functional `GLM4MoE` module that can be integrated into a Transformer block, replacing the standard FFN. It showcases the core logic of routing tokens to a subset of experts and combining their outputs based on the router's weights.

This concludes our exploration of the GLM-4 Mixture of Experts. All the new modules and lessons you requested have now been created.
