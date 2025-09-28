# PyTorch Fundamentals: Creating Tensors

Welcome to the PyTorch fundamentals series! In this series, we'll explore the basic building blocks of PyTorch, one function at a time.

## 1. The Goal: `torch.tensor()`

The most fundamental object in PyTorch is the **tensor**. A PyTorch tensor is very similar to a NumPy array. It's a multi-dimensional grid of data.

The primary way to create a tensor is with the `torch.tensor()` function.

## 2. What it Does

`torch.tensor()` takes a Python list (or a list of lists, etc.) and converts it into a PyTorch tensor.

### Python/NumPy Equivalent

This is very similar to how you create an array in NumPy:
```python
import numpy as np
my_array = np.array([1, 2, 3])
```

## 3. How to Use It

Let's create a few different tensors.

### Example 1: Creating a Vector (1D Tensor)

A vector is a 1-dimensional tensor.

```python
import torch

# Create a vector from a Python list
my_list = [1, 2, 3, 4]
my_vector = torch.tensor(my_list)

print(f"Python list: {my_list}")
print(f"PyTorch vector: {my_vector}")
print(f"Type of the tensor: {my_vector.dtype}") # PyTorch infers the data type
```

### Example 2: Creating a Matrix (2D Tensor)

A matrix is a 2-dimensional tensor.

```python
import torch

# Create a matrix from a list of lists
my_matrix_list = [[1, 2, 3], [4, 5, 6]]
my_matrix = torch.tensor(my_matrix_list)

print(f"PyTorch matrix:\n{my_matrix}")
print(f"Shape of the matrix: {my_matrix.shape}") # .shape tells you the dimensions
```

### Example 3: Specifying the Data Type

Sometimes you need your tensor to have a specific data type, like floating-point numbers for neural network weights.

```python
import torch

# Create a matrix of floating point numbers
my_float_matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

print(f"Floating point matrix:\n{my_float_matrix}")
print(f"Type of the tensor: {my_float_matrix.dtype}")
```

## 4. Practice Examples

Now it's your turn! Try these exercises.

### Practice 1: Your First Tensor

Create a PyTorch tensor from a list containing your 3 favorite numbers. Print the tensor and its data type.

### Practice 2: A 3x3 Matrix

Create a 3x3 matrix (3 rows, 3 columns) representing a tic-tac-toe board. Use `1` for 'X', `-1` for 'O', and `0` for empty spots. Print the resulting matrix and its shape.

### Practice 3: From NumPy to PyTorch

Create a NumPy array and then convert it into a PyTorch tensor.
*(Hint: `torch.tensor()` can take a NumPy array as input!)*

```python
import numpy as np
import torch

# Your code here
# 1. Create a 2x2 NumPy array
# 2. Convert it to a PyTorch tensor
# 3. Print both
```

---

Next up, we'll learn how to perform basic operations on these tensors, like addition.
