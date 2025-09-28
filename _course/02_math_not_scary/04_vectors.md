# Vectors: The Language of Data

## Table of Contents
1. [What are Vectors?](#what-are-vectors)
2. [Vector Operations](#vector-operations)
3. [Vectors in Python](#vectors-in-python)
4. [Vectors in PyTorch](#vectors-in-pytorch)
5. [Key Takeaways](#key-takeaways)
6. [Next Steps](#next-steps)

## What are Vectors?

A **vector** is a mathematical object that has both **magnitude** (size) and **direction**. You can think of it as an arrow pointing from a starting point to an ending point. In data science and machine learning, we often use vectors to represent data points.

### Geometric Interpretation
Geometrically, a vector is an arrow in a coordinate system. For example, in a 2D plane, a vector `v = [3, 2]` represents an arrow that starts at the origin (0,0) and ends at the point (3,2).

### Algebraic Representation
Algebraically, a vector is an ordered list of numbers. The numbers in the list are called the **components** or **elements** of the vector.

- A 2D vector: `v = [x, y]`
- A 3D vector: `v = [x, y, z]`
- An n-dimensional vector: `v = [x1, x2, ..., xn]`

### Simple Examples

#### Hand Calculation Examples

**Example 1: Representing a point in 2D space**

Imagine a point `P` with coordinates (4, 3). We can represent the position of this point with a vector `v` starting from the origin (0,0).

- Vector `v = [4, 3]`
- The magnitude (length) of the vector is calculated using the Pythagorean theorem: `||v|| = sqrt(4^2 + 3^2) = sqrt(16 + 9) = sqrt(25) = 5`
- The direction is the angle it makes with the positive x-axis.

**Example 2: Representing features of a house**

We can use a vector to represent the features of a house for a machine learning model:

- `house_vector = [number_of_bedrooms, square_footage, age_in_years]`
- A specific house could be: `house_1 = [3, 1500, 20]`

This is a 3-dimensional vector.

#### Code Examples

```python
import matplotlib.pyplot as plt

# Vector representation in Python
vector_v = [4, 3]

# Plotting the vector
plt.figure()
plt.quiver(0, 0, vector_v[0], vector_v[1], angles='xy', scale_units='xy', scale=1, color='r')
plt.xlim(0, 5)
plt.ylim(0, 4)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Geometric Representation of a Vector')
plt.grid()
plt.show()
```

## Vector Operations

We can perform several operations on vectors.

### 1. Vector Addition

To add two vectors, we add their corresponding components.

#### Hand Calculation Examples

**Example: `a = [1, 2]` and `b = [3, 1]`**

`a + b = [1+3, 2+1] = [4, 3]`

Geometrically, this is like placing the tail of vector `b` at the head of vector `a`. The resulting vector goes from the tail of `a` to the head of `b`.

### 2. Scalar Multiplication

To multiply a vector by a scalar (a single number), we multiply each component by that scalar.

#### Hand Calculation Examples

**Example: `a = [2, 3]` and `scalar s = 2`**

`s * a = 2 * [2, 3] = [2*2, 2*3] = [4, 6]`

This operation scales the vector, making it longer or shorter. If the scalar is negative, it reverses the vector's direction.

### 3. Dot Product

The dot product of two vectors is a scalar value. It is calculated by multiplying corresponding components and summing the results.

#### Hand Calculation Examples

**Example: `a = [1, 2, 3]` and `b = [4, 5, 6]`**

`a · b = (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32`

The dot product is related to the angle between two vectors. If the dot product is 0, the vectors are orthogonal (perpendicular).

## Vectors in Python

In Python, we commonly use lists or NumPy arrays to represent vectors. NumPy is highly recommended for numerical operations.

```python
import numpy as np

# Using NumPy arrays for vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vector Addition
c = a + b
print(f"a + b = {c}")

# Scalar Multiplication
s = 2
d = s * a
print(f"2 * a = {d}")

# Dot Product
dot_product = np.dot(a, b)
print(f"a · b = {dot_product}")

# Magnitude of a vector
magnitude_a = np.linalg.norm(a)
print(f"Magnitude of a = {magnitude_a}")
```

## Vectors in PyTorch

PyTorch is a popular deep learning library, and it uses **tensors** for all its operations. A vector is just a 1-dimensional tensor.

```python
import torch

# Using PyTorch tensors for vectors
a_torch = torch.tensor([1, 2, 3])
b_torch = torch.tensor([4, 5, 6])

# Vector Addition
c_torch = a_torch + b_torch
print(f"a + b (PyTorch) = {c_torch}")

# Scalar Multiplication
s_torch = 2
d_torch = s_torch * a_torch
print(f"2 * a (PyTorch) = {d_torch}")

# Dot Product
dot_product_torch = torch.dot(a_torch.float(), b_torch.float())
print(f"a · b (PyTorch) = {dot_product_torch}")

# Magnitude of a vector
magnitude_a_torch = torch.linalg.norm(a_torch.float())
print(f"Magnitude of a (PyTorch) = {magnitude_a_torch}")
```

## Key Takeaways

1.  **Vectors have magnitude and direction.**
2.  **In ML, vectors represent data points or features.**
3.  **Vector operations like addition, scalar multiplication, and dot product are fundamental.**
4.  **NumPy and PyTorch provide powerful tools for working with vectors in Python.**

## Next Steps

Now that you have a grasp of vectors, you can explore:
- **Matrices**: 2D arrays of numbers, which can be seen as collections of vectors.
- **Tensors**: The generalization of vectors and matrices to any number of dimensions.
- **Linear Transformations**: How matrices can operate on vectors to rotate, scale, and skew them.
