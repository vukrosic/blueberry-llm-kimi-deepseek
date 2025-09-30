# Activation Functions: Tanh

The Hyperbolic Tangent function, or Tanh, is a close relative of the Sigmoid function. It was also popular in earlier neural networks and addresses one of Sigmoid's key weaknesses.

## 1. The Goal: Understand Tanh

The Tanh function is defined as:
`tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)`

It takes any real number and squashes it to a value between **-1 and 1**.

## 2. Visualizing Tanh

The function's shape is very similar to Sigmoid, but it is centered at zero.

```python
import torch
import matplotlib.pyplot as plt

z = torch.linspace(-5, 5, 100)
output = torch.tanh(z)

plt.figure(figsize=(8, 4))
plt.plot(z, output)
plt.title("Tanh Activation Function")
plt.xlabel("Input (z)")
plt.ylabel("Activation Output")
plt.grid(True)
plt.show()
```

## 3. Tanh vs. Sigmoid

Tanh has one significant advantage over Sigmoid for use in hidden layers:

**It is zero-centered.**

The output of the Tanh function ranges from -1 to 1. This means the outputs from a Tanh-activated layer are, on average, closer to zero. This helps to center the inputs to the next layer, which can lead to faster convergence during training compared to the non-zero-centered Sigmoid.

## 4. Disadvantages

Despite this improvement, Tanh still suffers from the same **vanishing gradient problem** as Sigmoid.

The gradient of Tanh is `1 - tanh(z)²`. Similar to Sigmoid, this gradient becomes extremely small when the input `z` is large (positive or negative), causing gradients to vanish in deep networks.

For this reason, ReLU and its variants have almost completely replaced Tanh in the hidden layers of modern deep neural networks.

## 5. How to Use It

Using Tanh in PyTorch is simple.

```python
import torch

# A tensor of pre-activation scores
z = torch.tensor([-2.5, -0.1, 0.0, 1.5, 3.0])

# Apply Tanh
output = torch.tanh(z)

print(f"Original scores: {z}")
print(f"After Tanh:      {output}")
```

## 6. Practice Examples

### Practice 1: Compare the Ranges
Create a random tensor `z`. Apply both `torch.sigmoid(z)` and `torch.tanh(z)` to it. Print the minimum and maximum values of each output tensor to confirm their respective ranges (0 to 1 vs. -1 to 1).

### Practice 2: Relationship to Sigmoid
There is a direct mathematical relationship between Tanh and Sigmoid:
`tanh(z) = 2 * sigmoid(2z) - 1`

Verify this relationship. Write code that calculates the Tanh of a value `z` using only the `torch.sigmoid` function and basic arithmetic. Compare your result to the output of `torch.tanh(z)`.

---
While ReLU, Sigmoid, and Tanh are the classical activation functions, modern architectures often use more advanced functions. Next, we'll look at SiLU (also known as Swish).

---

**Next Lesson**: [SiLU](04_silu.md)
