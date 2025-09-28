# Activation Functions: SiLU (Swish)

The Sigmoid-weighted Linear Unit (SiLU), also known by its original name Swish, is a more modern activation function that has been shown to outperform ReLU in many deep learning models, particularly in computer vision.

## 1. The Goal: Understand SiLU

The SiLU function is defined as:
`SiLU(z) = z * σ(z)`

Where `σ(z)` is the standard Sigmoid function.

It multiplies the input `z` by the sigmoid of `z`. This creates a "gating" mechanism:
- When `z` is very negative, `σ(z)` is close to 0, so the output is close to 0.
- When `z` is very positive, `σ(z)` is close to 1, so the output is close to `z`.

## 2. Visualizing SiLU

The function looks like a smoothed-out version of ReLU.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

z = torch.linspace(-5, 5, 100)
# PyTorch's functional API has a silu function
output = F.silu(z)

plt.figure(figsize=(8, 4))
plt.plot(z, output)
plt.title("SiLU / Swish Activation Function")
plt.xlabel("Input (z)")
plt.ylabel("Activation Output")
plt.grid(True)
plt.show()
```

## 3. Properties of SiLU

SiLU combines some of the best features of ReLU and Sigmoid.

- **Non-Monotonic**: Unlike ReLU, SiLU dips slightly below zero for small negative values. This "bump" can help gradients push out of small negative values and may contribute to better learning.
- **Smoothness**: The function is smooth everywhere, which can lead to more stable training and better optimization compared to the sharp corner of ReLU.
- **Unbounded Above, Bounded Below**: Like ReLU, it is unbounded above, preventing saturation for large positive inputs. It is bounded below (at ≈ -0.28), which helps to regularize the network.
- **Self-Gating**: The `z * σ(z)` formulation is a form of "self-gating," where the function itself determines how much of the input `z` to pass through.

## 4. Disadvantages

The main disadvantage of SiLU compared to ReLU is that it is **more computationally expensive**. It involves an exponential function (`e⁻ᶻ`) inside the sigmoid, which is slower to compute than the simple `max(0, z)` of ReLU.

In many cases, the performance improvement in terms of model accuracy is worth the small computational overhead.

## 5. How to Use It

PyTorch provides a simple way to use SiLU.

```python
import torch
import torch.nn.functional as F

# A tensor of pre-activation scores
z = torch.tensor([-2.5, -0.1, 0.0, 1.5, 3.0])

# Apply SiLU
output = F.silu(z)

# Manual calculation for verification
manual_output = z * torch.sigmoid(z)

print(f"Original scores: {z}")
print(f"After SiLU:      {output}")
print(f"Manual calc:     {manual_output}")
```

## 6. Practice Examples

### Practice 1: The Negative Dip
Find the approximate minimum value of the SiLU function. At what input `z` does it occur? You can do this by creating a tensor of inputs from -2 to 0 and using `torch.min()`.

### Practice 2: Compare to ReLU
Create a tensor of inputs `z = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])`.
Calculate the output for both `F.relu(z)` and `F.silu(z)`.
What are the key differences you observe, especially for negative inputs?

---
SiLU is a powerful self-gated activation. Next, we'll look at SwiGLU, a more advanced variant that is becoming very popular in modern language models like LLaMA.
