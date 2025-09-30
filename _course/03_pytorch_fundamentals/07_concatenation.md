# PyTorch Fundamentals: Concatenating Tensors

Concatenation is the process of joining tensors together. This is a common operation when you need to combine data from different sources or merge feature maps in a neural network.

## 1. The Goal: `torch.cat()`

The main function for this operation is `torch.cat()`. It takes a sequence of tensors and joins them along a specified dimension.

## 2. What it Does

`torch.cat()` requires all tensors in the sequence to have the same shape, except in the dimension you are concatenating along.

Imagine you have two matrices, both of shape `(2, 3)`.
- If you concatenate them along `dim=0` (the row dimension), you will stack them vertically, resulting in a `(4, 3)` tensor.
- If you concatenate them along `dim=1` (the column dimension), you will join them side-by-side, resulting in a `(2, 6)` tensor.

### Python/NumPy Equivalent

NumPy has a `np.concatenate()` function that works in the exact same way.

```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# Concatenate along rows (dim=0)
C = np.concatenate((A, B), axis=0)
# C is [[1, 2], [3, 4], [5, 6], [7, 8]]
```

## 3. How to Use It

Let's see `torch.cat()` in action.

### Example 1: Stacking Vertically (along `dim=0`)

This is like adding more rows to your dataset.

```python
import torch

# Two matrices of the same shape (2x3)
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = torch.tensor([[7, 8, 9], [10, 11, 12]])

# Concatenate along dimension 0 (rows)
C = torch.cat((A, B), dim=0)

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Concatenated along dim=0 (shape {}):\n".format(C.shape), C)
```

### Example 2: Joining Side-by-Side (along `dim=1`)

This is like adding more features to your dataset.

```python
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = torch.tensor([[7, 8, 9], [10, 11, 12]])

# Concatenate along dimension 1 (columns)
D = torch.cat((A, B), dim=1)

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Concatenated along dim=1 (shape {}):\n".format(D.shape), D)
```

## 4. Practice Examples

### Practice 1: Combine Feature Sets

You have two sets of features for the same samples.
`features_1` is a `(10, 5)` tensor (10 samples, 5 features).
`features_2` is a `(10, 3)` tensor (the same 10 samples, 3 different features).

Combine these two tensors into a single `(10, 8)` tensor. Along which dimension should you concatenate?

### Practice 2: Build a Batch

You have four separate "images", each represented by a `(3, 32, 32)` tensor (channels, height, width).
Currently, they are in a Python list: `[image1, image2, image3, image4]`.

To feed this into a model, you need a single tensor of shape `(4, 3, 32, 32)`. This requires "stacking" them along a new dimension. `torch.stack()` is perfect for this. It's like `cat` but it adds a new dimension.

```python
import torch

# A list of 4 images
images = [torch.randn(3, 32, 32) for _ in range(4)]

# Use torch.stack to create a single batch tensor
batch = torch.stack(images, dim=0)

print(f"Shape of one image: {images[0].shape}")
print(f"Shape of the stacked batch: {batch.shape}")
```
Read the documentation for `torch.stack`. How is it different from `torch.cat`?

### Practice 3: Mismatched Shapes

Create two tensors, `X` of shape `(3, 4)` and `Y` of shape `(2, 4)`.
Try to concatenate them along `dim=1`. What happens? Why?
Now, try to concatenate them along `dim=0`. Does it work? Why?

---

Next, we'll look at creating tensors with pre-filled values like zeros and ones.
