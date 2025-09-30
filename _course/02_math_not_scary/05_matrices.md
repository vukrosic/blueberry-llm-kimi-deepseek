# Matrices: Organizing and Transforming Data

## Table of Contents
1. [What are Matrices?](#what-are-matrices)
2. [Matrix Operations](#matrix-operations)
3. [Matrices in Python with NumPy](#matrices-in-python-with-numpy)
4. [Matrices in PyTorch](#matrices-in-pytorch)
5. [Key Takeaways](#key-takeaways)
6. [Next Steps](#next-steps)

## What are Matrices?

A **matrix** is a rectangular grid of numbers, arranged in rows and columns. If you think of a vector as a single list of numbers (one row or one column), you can think of a matrix as a collection of vectors stacked together.

### A Simple Example: A Spreadsheet

The easiest way to think about a matrix is to imagine a spreadsheet or a table.

| Student ID | Exam 1 Score | Exam 2 Score |
|------------|--------------|--------------|
| 1          | 85           | 92           |
| 2          | 76           | 88           |
| 3          | 90           | 91           |

We can represent this table as a matrix `A`:

```
    [ 85  92 ]
A = [ 76  88 ]
    [ 90  91 ]
```

This is a **3x2 matrix** because it has 3 rows and 2 columns.

In machine learning, matrices are fundamental. They are used to store data (like the exam scores), represent transformations, and hold the "weights" of a neural network.

### Algebraic Representation

We denote a matrix by an uppercase letter (e.g., `A`). The dimensions are often written as `m x n`, where `m` is the number of rows and `n` is the number of columns.

- A 2x3 matrix:
```
    [ a b c ]
A = [ d e f ]
```

## Matrix Operations

Just like with vectors, we can perform operations on matrices.

### 1. Matrix Addition

To add two matrices, they must have the same dimensions. We simply add the corresponding elements.

#### Hand Calculation Example

**Example: `A = [[1, 2], [3, 4]]` and `B = [[5, 6], [7, 8]]`**

```
    [ 1+5  2+6 ]   [ 6  8 ]
A+B = [ 3+7  4+8 ] = [ 10 12 ]
```

### 2. Scalar Multiplication

To multiply a matrix by a scalar, we multiply every element in the matrix by that scalar.

#### Hand Calculation Example

**Example: `A = [[1, 2], [3, 4]]` and `scalar s = 3`**

```
      [ 3*1  3*2 ]   [ 3  6 ]
s * A = [ 3*3  3*4 ] = [ 9 12 ]
```

### 3. Matrix-Matrix Multiplication

Matrix multiplication is a bit more complex. To multiply matrix `A` (m x n) by matrix `B` (n x p), the number of columns in `A` must equal the number of rows in `B`. The resulting matrix `C` will have dimensions `m x p`.

The element at row `i`, column `j` of the result is the **dot product** of row `i` from `A` and column `j` from `B`.

#### Hand Calculation Example

**Example: `A` (2x2) and `B` (2x2)**

`A = [[1, 2], [3, 4]]`
`B = [[5, 6], [7, 8]]`

Result `C` will be 2x2.

- `C[0,0]` (1st row, 1st col) = (row 1 of A) · (col 1 of B) = (1*5) + (2*7) = 5 + 14 = 19
- `C[0,1]` (1st row, 2nd col) = (row 1 of A) · (col 2 of B) = (1*6) + (2*8) = 6 + 16 = 22
- `C[1,0]` (2nd row, 1st col) = (row 2 of A) · (col 1 of B) = (3*5) + (4*7) = 15 + 28 = 43
- `C[1,1]` (2nd row, 2nd col) = (row 2 of A) · (col 2 of B) = (3*6) + (4*8) = 18 + 32 = 50

So, `C = [[19, 22], [43, 50]]`

**Important**: Matrix multiplication is **not commutative**. In general, `A * B != B * A`.

## Matrices in Python with NumPy

NumPy is the standard library for numerical operations in Python and makes working with matrices easy.

```python
import numpy as np

# Create NumPy matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:\n", A)
print("Matrix B:\n", B)

# Matrix Addition
C = A + B
print("\nA + B:\n", C)

# Scalar Multiplication
s = 3
D = s * A
print("\n3 * A:\n", D)

# Matrix Multiplication
E = np.dot(A, B) # or A @ B
print("\nA * B:\n", E)
```

## Matrices in PyTorch

In PyTorch, matrices are represented as 2-dimensional tensors.

```python
import torch

# Create PyTorch tensors (matrices)
A_torch = torch.tensor([[1, 2], [3, 4]])
B_torch = torch.tensor([[5, 6], [7, 8]])

print("PyTorch Matrix A:\n", A_torch)
print("PyTorch Matrix B:\n", B_torch)

# Matrix Addition
C_torch = A_torch + B_torch
print("\nA + B (PyTorch):\n", C_torch)

# Scalar Multiplication
s_torch = 3
D_torch = s_torch * A_torch
print("\n3 * A (PyTorch):\n", D_torch)

# Matrix Multiplication
E_torch = torch.matmul(A_torch, B_torch) # or A_torch @ B_torch
print("\nA * B (PyTorch):\n", E_torch)
```

## Key Takeaways

1.  **Matrices are grids of numbers**, useful for storing and organizing data.
2.  The dimensions of a matrix are `rows x columns`.
3.  Basic operations include addition, scalar multiplication, and matrix multiplication.
4.  **Matrix multiplication is not commutative (`A @ B != B @ A`)**.
5.  NumPy and PyTorch are the standard tools for matrix operations in Python for data science and deep learning.

## Next Steps

Understanding matrices opens the door to more advanced topics:
- **Linear Transformations**: How matrices can "transform" vectors (rotate, scale, shear). This is the foundation of computer graphics and neural network layers.
- **Tensors**: The generalization of matrices to more than two dimensions.
- **Eigenvalues and Eigenvectors**: The "special" vectors of a matrix that only get scaled by a transformation.

---

**Next Lesson**: [Creating Tensors](../03_pytorch_fundamentals/01_creating_tensors.md) (PyTorch Module)

