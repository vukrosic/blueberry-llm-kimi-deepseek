# PyTorch Fundamentals: Matrix Multiplication

Matrix multiplication is one of the most important operations in deep learning. It's how data is transformed as it passes through the layers of a neural network.

## 1. The Goal: `torch.matmul()` and `@`

Similar to addition, there are two ways to perform matrix multiplication in PyTorch:
1.  Using the `torch.matmul()` function.
2.  Using the Python `@` operator (this is the standard for matrix multiplication).

## 2. What it Does

Matrix multiplication is **not** element-wise multiplication. It's a more complex operation.

To multiply matrix `A` (with shape `m x n`) by matrix `B` (with shape `n x p`), the number of columns in `A` **must equal** the number of rows in `B`. The resulting matrix `C` will have the shape `m x p`.

The value at `C[i, j]` is the **dot product** of the i-th row of `A` and the j-th column of `B`.

### Python/NumPy Equivalent

NumPy also uses the `@` operator or the `np.matmul()` function.
```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B
# C is [[19, 22], [43, 50]]
```

## 3. How to Use It

Let's multiply two matrices.

### Example 1: Using the `@` operator

This is the modern, preferred way to do matrix multiplication.

```python
import torch

# Create two matrices with compatible shapes
# A is 2x3, B is 3x2
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = torch.tensor([[7, 8], [9, 10], [11, 12]])

# Multiply them. The result will be a 2x2 matrix.
C = A @ B

print("Matrix A (2x3):\n", A)
print("Matrix B (3x2):\n", B)
print("A @ B (2x2):\n", C)
```
*Self-check: Calculate `C[0,0]` by hand. It's `(1*7) + (2*9) + (3*11) = 7 + 18 + 33 = 58`.*

### Example 2: Using `torch.matmul()`

This function behaves identically to the `@` operator.

```python
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = torch.tensor([[7, 8], [9, 10], [11, 12]])

# Multiply them using the function
C = torch.matmul(A, B)

print("torch.matmul(A, B):\n", C)
```

### Important: Order Matters!

Unlike regular multiplication or element-wise addition, **matrix multiplication is not commutative**. This means `A @ B` is not the same as `B @ A`.

If you try to calculate `B @ A` in our example, it will work, but the result will be a 3x3 matrix with different values.

```python
# B (3x2) @ A (2x3) results in a 3x3 matrix
D = B @ A
print("B @ A (3x3):\n", D)
```

## 4. Practice Examples

### Practice 1: Shape Mismatch

Create two matrices, `X` (2x3) and `Y` (2x2). Try to multiply them (`X @ Y`). What happens? Why?

### Practice 2: Transform a Vector

In machine learning, we often multiply a matrix by a vector to "transform" it.
Create a matrix `M` of shape 2x2 and a vector `v` of shape 2 (which is treated as 2x1).
Multiply `M` by `v`. What is the shape of the resulting vector?

```python
import torch

M = torch.tensor([[2, 0], [0, 2]]) # This matrix scales things by 2
v = torch.tensor([5, 10])

# Your code here
# result = M @ v
# print(result)
```

### Practice 3: Element-wise vs. Matrix Multiplication

Create two 2x2 matrices, `A` and `B`.
Calculate `C1 = A * B` (element-wise multiplication).
Calculate `C2 = A @ B` (matrix multiplication).
Print both results. Notice how different they are!

---

Next, we'll learn how to change the "view" of a tensor by transposing it.

---

**Next Lesson**: [Transposing](04_transposing.md)
