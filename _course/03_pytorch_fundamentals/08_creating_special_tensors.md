# PyTorch Fundamentals: Creating Special Tensors

Often, you need to create tensors with specific initial values, such as all zeros or all ones. This is common for initializing weights or creating masks. PyTorch has several factory functions for this purpose.

## 1. The Goal: `torch.zeros()`, `torch.ones()`, `torch.randn()`

We will explore the most common functions for creating tensors with pre-filled data:
- `torch.zeros()`: Creates a tensor filled with `0`.
- `torch.ones()`: Creates a tensor filled with `1`.
- `torch.randn()`: Creates a tensor filled with random numbers from a standard normal distribution (mean 0, variance 1).

## 2. What They Do

These functions create a new tensor with a specified shape, filled with the corresponding values.

### Python/NumPy Equivalent

NumPy has identical functions: `np.zeros()`, `np.ones()`, and `np.random.randn()`.

```python
import numpy as np
# A 2x3 matrix of zeros
zeros_array = np.zeros((2, 3))
# A 1D array of ones
ones_array = np.ones(5)
```

## 3. How to Use It

The first argument to these functions is always the desired shape of the tensor.

### Example 1: `torch.zeros()`

This is often used to create a blank slate tensor that will be filled in later, or for biases that start at zero.

```python
import torch

# Create a 2x3 matrix of zeros
zeros_tensor = torch.zeros(2, 3)

print(f"Zeros tensor of shape {zeros_tensor.shape}:\n{zeros_tensor}")
```

### Example 2: `torch.ones()`

This can be useful for creating masks or for certain mathematical initializations.

```python
import torch

# Create a vector of 4 ones
ones_tensor = torch.ones(4)

print(f"Ones tensor of shape {ones_tensor.shape}:\n{ones_tensor}")
```

### Example 3: `torch.randn()`

This is one of the most common ways to initialize the weights of a neural network, as it breaks symmetry and provides a good starting point for optimization.

```python
import torch

# Create a 3x2 matrix of random numbers
random_tensor = torch.randn(3, 2)

print(f"Random tensor of shape {random_tensor.shape}:\n{random_tensor}")
```

### Bonus: `_like()` functions

Sometimes you want to create a new tensor that has the exact same shape as another tensor. The `_like` functions are perfect for this.

```python
import torch

# An existing tensor
A = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Create a tensor of zeros with the same shape as A
zeros_like_A = torch.zeros_like(A)

print(f"Original shape: {A.shape}")
print(f"zeros_like shape: {zeros_like_A.shape}")
print(f"zeros_like tensor:\n{zeros_like_A}")
```

## 4. Practice Examples

### Practice 1: Create a Mask

Create a 5x5 matrix of all zeros, but then use indexing to set the inner 3x3 matrix to all ones. This is a common way to create a "mask".

### Practice 2: Initialize a Bias Vector

In a neural network, a bias is typically a vector (1D tensor) initialized to zeros. Create a bias vector for a layer that has 128 neurons.

### Practice 3: Random Weight Matrix

Create a weight matrix for a linear layer that takes 784 input features and produces 256 output features. The shape should be `(784, 256)`. Initialize it with random numbers from a standard normal distribution.

---

This concludes our extended tour of PyTorch fundamentals! You now have the tools to create, manipulate, and combine tensors. Next, we'll see how to build a neuron from scratch using these concepts.
