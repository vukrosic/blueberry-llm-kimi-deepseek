# Vectors: The Language of Data

## Table of Contents
1. [What are Vectors?](#what-are-vectors)
2. [Vector Operations](#vector-operations)
3. [Vectors in Python](#vectors-in-python)
4. [Vectors in PyTorch](#vectors-in-pytorch)
5. [Key Takeaways](#key-takeaways)
6. [Next Steps](#next-steps)

## What are Vectors?

Let's start with a simple example. Imagine you're giving directions to a friend. You might say, "Walk 4 blocks east and 3 blocks north." This instruction is a **vector**. It's a package of information that contains both **distance** (how far) and **direction** (where to go).

In mathematics and data science, a vector is simply an ordered list of numbers. We can write the instruction above as:
`directions = [4, 3]`

We often use vectors to represent data. For instance, the features of a house could be a vector:
`house_features = [number_of_bedrooms, square_footage, age_in_years]`
A specific house might be represented as:
`house_1 = [3, 1500, 20]`

### The Two Key Properties of a Vector

Every vector has two key properties:

1.  **Magnitude**: This is the "size" or "length" of the vector. For our walking directions `[4, 3]`, the magnitude is the total straight-line distance from the start to the end point. We can calculate it using the Pythagorean theorem.
    - `Magnitude = sqrt(4^2 + 3^2) = sqrt(16 + 9) = sqrt(25) = 5` blocks.

2.  **Direction**: This is the way the vector is pointing. Geometrically, you can think of it as an arrow.

### Visualizing Vectors

We can visualize vectors as arrows in a coordinate system. The vector `v = [4, 3]` would be an arrow starting from the origin (0,0) and ending at the point (4,3).

```python
import matplotlib.pyplot as plt

# The vector from our example
vector_v = [4, 3]

# Plotting the vector
plt.figure()
# Create an arrow from (0,0) to (4,3)
plt.quiver(0, 0, vector_v[0], vector_v[1], angles='xy', scale_units='xy', scale=1, color='r')
# Set the plot limits
plt.xlim(0, 5)
plt.ylim(0, 4)
# Add labels and a grid
plt.xlabel('East-West Blocks')
plt.ylabel('North-South Blocks')
plt.title('Visualizing the Direction Vector [4, 3]')
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
