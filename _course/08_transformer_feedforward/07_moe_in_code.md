# Lesson 7: MoE in Code

Let's put together a conceptual implementation of a Mixture of Experts (MoE) layer. This will show how the gating network and the experts work together.

We'll use Python and NumPy to build a simple MoE layer that selects the top 1 expert.

```python
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# A simple FFN for our experts
class Expert:
    def __init__(self, input_dim, hidden_dim):
        self.w1 = np.random.randn(input_dim, hidden_dim)
        self.w2 = np.random.randn(hidden_dim, input_dim)

    def forward(self, x):
        # Simple ReLU activation
        hidden = np.maximum(0, np.dot(x, self.w1))
        return np.dot(hidden, self.w2)

class MoELayer:
    def __init__(self, input_dim, hidden_dim, num_experts):
        self.num_experts = num_experts
        self.experts = [Expert(input_dim, hidden_dim) for _ in range(num_experts)]
        
        # The gating network is just a linear layer
        self.gate = np.random.randn(input_dim, num_experts)

    def forward(self, x):
        # 1. Calculate gating scores
        # (batch_size, num_experts)
        gating_logits = np.dot(x, self.gate)
        gating_weights = softmax(gating_logits)

        # --- Top-k Gating (simplified to k=1) ---
        # Find the index of the best expert for each token
        best_expert_indices = np.argmax(gating_weights, axis=1)

        # 2. Get expert outputs
        final_output = np.zeros_like(x)
        
        # This loop is for demonstration. In a real implementation,
        # this would be vectorized and much more complex.
        for i, token_input in enumerate(x):
            expert_idx = best_expert_indices[i]
            selected_expert = self.experts[expert_idx]
            
            # Get the output from the selected expert
            expert_output = selected_expert.forward(token_input)
            
            # Weight the output by the gating score
            # (In top-1, this weight is 1, but we show it for clarity)
            gating_score = gating_weights[i, expert_idx]
            final_output[i] = gating_score * expert_output

        return final_output

# --- Example Usage ---

# Batch of 4 tokens, each with dimension 10
input_data = np.random.randn(4, 10)

moe_layer = MoELayer(input_dim=10, hidden_dim=20, num_experts=8)
output = moe_layer.forward(input_data)

print("Output shape:", output.shape)

```

## What the Code is Doing

1.  **Expert Class:** We define a simple `Expert` class, which is just an FFN.

2.  **MoELayer Class:**
    -   In the `__init__` method, we create a list of `Expert` objects and the gating network's weight matrix.
    -   In the `forward` method, we first calculate the `gating_weights` by passing the input through the gate.
    -   We then use `np.argmax` to find the single best expert for each token in the batch (our simplified top-1 gating).
    -   We then loop through each token, pass it to its selected expert, and apply the gating score.

## Important Note on Real Implementations

This code is a simplified demonstration. Real-world MoE implementations (like in PyTorch or TensorFlow) are much more complex. They use sophisticated techniques to handle the routing of tokens to different experts in a way that is efficient on GPUs, which is a major engineering challenge.

## Conclusion

You have now seen how a Feed-Forward Network and a Mixture of Experts layer work, from the high-level concept down to a code implementation. These components are essential for the impressive capabilities of modern large language models.
