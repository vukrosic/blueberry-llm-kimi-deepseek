# PyTorch Fundamentals: Reshaping Tensors

Reshaping is a powerful tool that allows you to change the shape of a tensor without changing its data. This is crucial when you need to feed data into a neural network layer that expects a specific input shape.

## 1. The Goal: `torch.reshape()` and `.view()`

The two main ways to reshape a tensor are:
1.  `torch.reshape(input, shape)` or `tensor.reshape(shape)`
2.  `tensor.view(shape)`

They are very similar, but have a subtle difference. For most purposes, `reshape` is recommended.

## 2. What it Does

Reshaping changes the dimensions of a tensor, as long as the total number of elements remains the same. For example, a tensor with 12 elements can be shaped as `(12)`, `(2, 6)`, `(6, 2)`, `(3, 4)`, `(4, 3)`, `(2, 2, 3)`, etc.

- A `(2, 6)` tensor (2 rows, 6 columns) has `2 * 6 = 12` elements.
- A `(2, 2, 3)` tensor has `2 * 2 * 3 = 12` elements.

### Python/NumPy Equivalent

NumPy has a nearly identical `np.reshape()` function.
```python
import numpy as np
a = np.arange(12) # array([0, 1, ..., 11])
b = a.reshape(3, 4)
# b is now a 3x4 matrix
```

## 3. How to Use It

Let's flatten a matrix into a vector and then build it back up.

### Example 1: Using `reshape()`

This is the most flexible and generally recommended method.

```python
import torch

# Create a 2x3 matrix
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Original shape: {A.shape}")

# Reshape it into a 1D vector of length 6
B = A.reshape(6)
print(f"Reshaped to (6): {B}")
print(f"New shape: {B.shape}")

# Reshape it into a 3x2 matrix
C = A.reshape(3, 2)
print(f"Reshaped to (3, 2):\n{C}")
print(f"New shape: {C.shape}")
```

### Example 2: The `-1` Trick

Often, you want to reshape a dimension without having to calculate its size manually. You can use `-1` for one of the dimensions, and PyTorch will automatically infer the correct size.

```python
import torch

# Create a tensor with 12 elements
A = torch.arange(12) # Like range(12)

# Reshape to 4 rows, and automatically figure out the number of columns
B = A.reshape(4, -1)

print(f"Original tensor: {A}")
print(f"Reshaped to (4, -1):\n{B}")
print(f"Inferred shape: {B.shape}") # PyTorch figured out it should be 4x3
```
This is extremely common when preparing data for a model. For example, flattening a batch of images.

### Example 3: Using `.view()`

`.view()` works almost identically to `reshape`. The main difference is that `.view()` only works on tensors that are "contiguous" in memory (stored in a continuous block). `reshape` will automatically handle non-contiguous tensors by creating a copy if needed.

For beginners, the distinction is minor. **When in doubt, use `reshape`**.

```python
import torch

A = torch.arange(12)

# .view() works just like reshape here
B = A.view(3, 4)
print(f"Reshaped with .view(3, 4):\n{B}")
```

## 4. Practice Examples

### Practice 1: Flatten an Image

Imagine you have a single grayscale image represented as a 28x28 tensor. Many neural network layers expect a flat vector as input. Reshape this "image" into a single 1D vector. What is its length?

```python
import torch

# A fake 28x28 image
image = torch.randn(28, 28)

# Your code here
# flattened_image = image.reshape(-1)
# print(f"Original shape: {image.shape}")
# print(f"Flattened shape: {flattened_image.shape}")
```

### Practice 2: Add a "Batch" Dimension

You have a single image of shape `(28, 28)`. But PyTorch models expect a "batch" of images, with the shape `(batch_size, height, width)`.

Use `reshape` to add a batch dimension of 1 to your image tensor. The final shape should be `(1, 28, 28)`.


### Practice 3: `view` vs. `reshape`

This is a more advanced question. Let's see where `.view()` can fail.

```python
import torch

# Create a tensor
A = torch.arange(12).reshape(3, 4)

# Create a transposed version. Transposing can make a tensor non-contiguous.
A_t = A.T

# This will likely throw an error!
# A_t.view(12)

# But this will work!
# A_t.reshape(12)
```
Run the code above. Why does `.view()` fail while `.reshape()` succeeds? (Answer: Because `A_t` is not contiguous in memory, and `reshape` handles this by making a copy, whereas `view` requires it to be contiguous).

---

This concludes our initial tour of fundamental PyTorch tensor manipulations!

---

**Next Lesson**: [Indexing and Slicing](06_indexing_and_slicing.md)
