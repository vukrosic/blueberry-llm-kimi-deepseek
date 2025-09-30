# Activation Functions: SwiGLU

SwiGLU is a sophisticated activation function that has become a key component in modern large language models (LLMs) like LLaMA and Mixtral. It is a variant of "Gated Linear Units" (GLU).

## 1. The Goal: Understand SwiGLU

Unlike the previous activation functions, SwiGLU is not applied to a single tensor. It operates on **two tensors** and involves a split in the network's architecture.

The SwiGLU operation is defined as:
`SwiGLU(x, W, V, b, c) = SiLU(xW + b) ⊗ (xV + c)`

Let's break this down into a more understandable, practical implementation:

1.  An input tensor `x` is fed into a linear layer that outputs a tensor with **twice the number of dimensions** as we actually need.
2.  This larger output tensor is **split in half** into two new tensors, `A` and `B`.
3.  The SwiGLU activation is the **element-wise product** of `A` and the `SiLU` of `B`.

`Output = A * SiLU(B)`

## 2. The Gating Mechanism

This structure creates a powerful **gating mechanism**.
- The `SiLU(B)` part acts as the "gate". The SiLU function squashes the values in `B`, and for negative values, it outputs zero or near-zero.
- The `A` part contains the actual values or "content".
- When `A` is multiplied by the gate, some elements of `A` are allowed to pass through (where the gate is close to 1), while others are "turned off" or suppressed (where the gate is close to 0).

This allows the network to dynamically control which information flows through the layer, making it much more expressive than a simple point-wise activation like ReLU.

## 3. How to Implement It

Let's implement a SwiGLU module in PyTorch. This is typically done as part of a "Feed-Forward" block in a Transformer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    A SwiGLU activation module.
    This is typically used inside a feed-forward network block.
    """
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        # If hidden_features is not provided, a common practice is to
        # set it to 2/3 of 4 times the input features, and make it
        # divisible by a certain number (e.g., 256) for efficiency.
        # Here, we'll just use a simpler default for clarity.
        hidden_features = hidden_features or in_features * 2

        # The single linear layer that projects to twice the hidden dimension
        self.linear_up = nn.Linear(in_features, hidden_features * 2)

    def forward(self, x):
        # 1. Project to twice the hidden dimension
        # Input x shape: (batch, seq_len, in_features)
        # projected_x shape: (batch, seq_len, hidden_features * 2)
        projected_x = self.linear_up(x)

        # 2. Split the tensor in half
        # A and B will both have shape: (batch, seq_len, hidden_features)
        A, B = torch.chunk(projected_x, 2, dim=-1)

        # 3. Apply the SwiGLU activation
        # The result has shape: (batch, seq_len, hidden_features)
        return A * F.silu(B)

# --- Example Usage ---

# Configuration
batch_size = 4
seq_len = 10
in_features = 32
hidden_features = 64 # The desired output dimension of the block

# Create some random input data
x = torch.randn(batch_size, seq_len, in_features)

# Create and apply the SwiGLU module
swiglu_layer = SwiGLU(in_features, hidden_features)
output = swiglu_layer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```
Notice that the output dimension (`hidden_features`) is different from the input dimension (`in_features`). A complete feed-forward block in a Transformer would include another linear layer to project this back down to `in_features`.

## 4. Advantages and Disadvantages

- **Advantage**: Greatly improved expressiveness and performance, which is why it's used in state-of-the-art LLMs.
- **Disadvantage**: Increased parameter count. The `linear_up` layer has to project to `hidden_features * 2`, which requires more parameters than a standard feed-forward layer. However, research has shown this is a worthwhile trade-off.

---
Finally, we'll look at Softmax, the standard activation function for multi-class classification problems.

---

**Next Lesson**: [Softmax](06_softmax.md)
