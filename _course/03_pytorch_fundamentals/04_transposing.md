# PyTorch Fundamentals: Transposing Tensors

Transposing is a fundamental operation that "flips" a tensor over its diagonal. This swaps the row and column dimensions. It's often used to make tensor shapes compatible for operations like matrix multiplication.

## 1. The Goal: `torch.transpose()`, `.t()`, and `.T`

There are a few ways to transpose a tensor in PyTorch:
1.  `torch.transpose(input, dim0, dim1)`: The most general function.
2.  `.t()`: A convenient method for 2D tensors (matrices).
3.  `.T`: A property that acts like a function for tensors of any dimension.

## 2. What it Does

Transposing a matrix (a 2D tensor) swaps its rows and columns. If a matrix has a shape of `m x n`, its transpose will have a shape of `n x m`.

### Python/NumPy Equivalent

NumPy uses the same `.T` property.
```python
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]]) # Shape 2x3
B = A.T
# B is [[1, 4], [2, 5], [3, 6]] with shape 3x2
```

## 3. How to Use It

Let's transpose a matrix.

### Example 1: Using the `.T` property

This is often the quickest and most readable method.

```python
import torch

# Create a 2x3 matrix
A = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Get its transpose
A_t = A.T

print("Original Matrix (2x3):\n", A)
print("Transposed Matrix (3x2):\n", A_t)
```

### Example 2: Using `.t()` for 2D Tensors

The `.t()` method is a convenient shorthand specifically for 2D tensors.

```python
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Transpose it
A_t = A.t()

print("Original Matrix:\n", A)
print("Transposed with .t():\n", A_t)
```

### Example 3: Using `torch.transpose()`

This function is more general and powerful because you can specify which dimensions to swap. For a 2D tensor, you swap dimension 0 (rows) and dimension 1 (columns).

```python
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Transpose by swapping dimensions 0 and 1
A_t = torch.transpose(A, 0, 1)

print("Original Matrix:\n", A)
print("Transposed with torch.transpose():\n", A_t)
```

## 4. Practice Examples

### Practice 1: Fix a Shape Mismatch

Remember the shape mismatch error from the matrix multiplication lesson? Let's fix it with a transpose.

Create two matrices, `X` (2x3) and `Y` (2x2). You can't multiply them (`X @ Y`).
But what if you transpose `X` first?
Calculate `X.T @ Y`. What are the shapes? Does it work? Why?

```python
import torch

X = torch.randn(2, 3)
Y = torch.randn(2, 2)

# Your code here
# X_t = X.T
# result = X_t @ Y
# print(f"Shape of X.T: {X_t.shape}")
# print(f"Shape of Y: {Y.shape}")
# print(f"Shape of result: {result.shape}")
```

### Practice 2: Double Transpose

Create a matrix `A`. Calculate its transpose, `A_t`. Now, calculate the transpose of `A_t`. What do you get?

### Practice 3: Transposing a Batch of Matrices

In deep learning, you often work with batches of data. You might have a tensor of shape `(batch_size, rows, cols)`, for example, `(10, 3, 4)`. This represents 10 matrices, each 3x4.

Create a random tensor with this shape. Use `torch.transpose()` to swap the last two dimensions (the rows and columns). What is the new shape of the tensor?

```python
import torch

# Batch of 10 matrices, each 3x4
batch_A = torch.randn(10, 3, 4)

# Your code here
# Use torch.transpose to swap dimensions 1 and 2
# batch_A_t = torch.transpose(batch_A, 1, 2)
# print(f"Original shape: {batch_A.shape}")
# print(f"New shape: {batch_A_t.shape}")
```

---

Next, we'll learn how to arbitrarily reshape tensors into new configurations.
