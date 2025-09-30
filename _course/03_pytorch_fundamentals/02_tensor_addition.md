# PyTorch Fundamentals: Tensor Addition

Now that we know how to create tensors, let's learn how to perform a basic operation: addition.

## 1. The Goal: `torch.add()` and `+`

There are two common ways to add tensors in PyTorch:
1.  Using the `torch.add()` function.
2.  Using the standard Python `+` operator.

Both achieve the same result: element-wise addition.

## 2. What it Does

Element-wise addition means that each element in the first tensor is added to the corresponding element in the second tensor. For this to work, the tensors must have the same shape (or be "broadcastable", a topic for a later lesson!). 

### Python/NumPy Equivalent

This is identical to how addition works in NumPy.
```python
import numpy as np
a = np.array([1, 2])
b = np.array([3, 4])
c = a + b # Result is [4, 6]
```

## 3. How to Use It

Let's add two matrices.

### Example 1: Using the `+` operator

This is the most common and intuitive way to add tensors.

```python
import torch

# Create two matrices (2D tensors)
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[10, 20], [30, 40]])

# Add them together
C = A + B

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("A + B:\n", C)
```

### Example 2: Using `torch.add()`

The `torch.add()` function does the exact same thing.

```python
import torch

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[10, 20], [30, 40]])

# Add them using the function
C = torch.add(A, B)

print("torch.add(A, B):\n", C)
```

### Example 3: Addition with a Scalar

You can also add a single number (a scalar) to every element in a tensor.

```python
import torch

A = torch.tensor([[1, 2], [3, 4]])
scalar = 100

# Add the scalar to the tensor
C = A + scalar

print("Matrix A:\n", A)
print("A + 100:\n", C)
```

## 4. Practice Examples

Time to practice your tensor addition skills.

### Practice 1: Add Two Vectors

Create two vectors (1D tensors) of the same length and add them together.

### Practice 2: Update a Scoreboard

Imagine you have a tensor representing scores for 3 players over 2 rounds.
`scores = torch.tensor([[80, 85], [90, 95], [70, 75]])`

Now, imagine each player gets a 5-point bonus. Create a second tensor (or use a scalar) to add 5 points to every score. Print the original scores and the new scores.

### Practice 3: In-place Addition

PyTorch has a special method for in-place addition: `add_()`. This modifies the tensor directly instead of creating a new one.

```python
import torch

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[10, 20], [30, 40]])

# This will change A directly
A.add_(B)

print("A after in-place addition:\n", A)
```
Try this yourself. Create two tensors, `X` and `Y`, and add `Y` to `X` in-place. What happens to the value of `X`?

---

Next, we'll move on to a more complex operation: matrix multiplication.

---

**Next Lesson**: [Matrix Multiplication](03_matrix_multiplication.md)
