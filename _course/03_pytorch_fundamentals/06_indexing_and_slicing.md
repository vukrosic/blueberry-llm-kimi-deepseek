# PyTorch Fundamentals: Indexing and Slicing

Indexing and slicing are how you access and select parts of a tensor. This is a critical skill for manipulating data, accessing model weights, and selecting training examples.

## 1. The Goal: Accessing Tensor Data

We will use Python's familiar square-bracket `[]` notation to select data from tensors.

## 2. What it Does

Just like with Python lists or NumPy arrays, you can select specific elements, rows, columns, or even sub-tensors.

- **Indexing**: Accessing a single element using its position (e.g., `my_tensor[0]`).
- **Slicing**: Selecting a range of elements (e.g., `my_tensor[1:4]`).

### Python/NumPy Equivalent

The syntax is virtually identical to NumPy array indexing and slicing.

```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
# Get the first row
first_row = a[0, :]
# Get the element at row 1, column 2
element = a[1, 2] # Result is 6
```

## 3. How to Use It

Let's work with a 2D tensor (a matrix).

```python
import torch

# A 3x4 matrix
A = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
```

### Example 1: Selecting a Single Element

To get the number `7`, we need the element at row `1`, column `2`.

```python
# Remember, indexing is 0-based
element = A[1, 2]
print(f"Element at A[1, 2]: {element}")
```

### Example 2: Selecting a Row

To get the entire second row (`[5, 6, 7, 8]`), we specify the row index and use `:` for the column index to select all columns.

```python
row_1 = A[1, :]
# You can also just use one index for rows
# row_1_shorthand = A[1]

print(f"Second row: {row_1}")
```

### Example 3: Selecting a Column

To get the third column (`[3, 7, 11]`), we use `:` for the rows and specify the column index.

```python
col_2 = A[:, 2]
print(f"Third column: {col_2}")
```

### Example 4: Slicing a Sub-Matrix

Let's select a 2x2 sub-matrix from the top right: `[[3, 4], [7, 8]]`.
We need rows 0 and 1, and columns 2 and 3.

```python
# Select rows 0 up to (but not including) 2
# Select columns 2 up to (but not including) 4
sub_matrix = A[0:2, 2:4]

print(f"Sub-matrix:\n{sub_matrix}")
```

## 4. Practice Examples

### Practice 1: Get the Corners

Using the matrix `A` from our examples, write code to select the four corner elements: `1`, `4`, `9`, and `12`.

### Practice 2: The Inner Matrix

From a 4x4 matrix, select the inner 2x2 matrix. For example, if you have:
```
[[ 1,  2,  3,  4],
 [ 5,  6,  7,  8],
 [ 9, 10, 11, 12],
 [13, 14, 15, 16]]
```
Your code should select:
```
[[ 6,  7],
 [10, 11]]
```

### Practice 3: Select a Batch

In machine learning, data is often stored in a tensor of shape `(batch_size, features)`. 
Create a tensor of shape `(10, 5)` with random data. This represents a batch of 10 samples, each with 5 features.

Write code to select the first 3 samples from the batch. What is the shape of the resulting tensor?

---

Next, we'll see how to combine tensors.

---

**Next Lesson**: [Concatenation](07_concatenation.md)

```