# Neuron From Scratch: The Linear Step

In the last lesson, we saw that a neuron's calculation has two parts. Here, we'll focus on the first part: the **linear step**. This is the heart of the neuron's calculation, where it weighs the evidence.

## 1. The Goal: Calculate the Neuron's Score

The linear step combines the inputs and weights to produce a single number, which we called the "score". This score is also known as the **logit** or the **pre-activation output**.

The formula is:
`z = (w₁x₁ + w₂x₂ + ... + wₙxₙ) + b`

Where:
- `z` is the score (logit)
- `xᵢ` is the i-th input
- `wᵢ` is the weight for the i-th input
- `b` is the bias

## 2. Vectorization: The Language of Deep Learning

Doing this calculation one element at a time is slow and inefficient. In deep learning, we do everything with **vectors and matrices**.

Let's rewrite the formula using vectors:
- The inputs can be a vector `x = [x₁, x₂, ..., xₙ]`
- The weights can be a vector `w = [w₁, w₂, ..., wₙ]`

Now, the sum of products `(w₁x₁ + w₂x₂ + ...)` is simply the **dot product** of the input vector and the weight vector!

`z = w · x + b`

This is much cleaner and computationally much faster.

## 3. How to Implement It

Let's use PyTorch to perform this calculation.

### Example: A Neuron with 3 Inputs

Imagine a neuron with 3 inputs and a specific set of weights and a bias.

```python
import torch

# Inputs to the neuron
inputs = torch.tensor([0.5, 0.1, 0.8], dtype=torch.float32)

# Weights for each input
weights = torch.tensor([0.7, -0.3, 0.5], dtype=torch.float32)

# The neuron's bias
bias = torch.tensor(1.5, dtype=torch.float32)
```

Now, let's calculate the score `z`.

```python
# Using the dot product
score = torch.dot(inputs, weights) + bias

print(f"Inputs: {inputs}")
print(f"Weights: {weights}")
print(f"Bias: {bias}")
print(f"---")
print(f"Score (logit): {score}")
```

Let's verify the calculation by hand:
`Score = (0.5 * 0.7) + (0.1 * -0.3) + (0.8 * 0.5) + 1.5`
`Score = 0.35 - 0.03 + 0.4 + 1.5`
`Score = 2.22`

The code should give us the same result!

## 4. Processing a Batch of Data

A neural network almost never processes a single input at a time. It processes a **batch** of inputs for efficiency.

Instead of a single input vector, we now have a matrix of inputs.
- `Inputs` shape: `(batch_size, num_features)`
- `Weights` shape: `(num_features)`
- `Bias` is still a single number.

We can't take the dot product of a matrix and a vector directly in this way. Instead, we use **matrix-vector multiplication**.

```python
import torch

# A batch of 2 inputs, each with 3 features
inputs_batch = torch.tensor([
    [0.5, 0.1, 0.8],  # First input
    [0.9, 0.2, 0.1]   # Second input
], dtype=torch.float32)

# Weights and bias remain the same
weights = torch.tensor([0.7, -0.3, 0.5], dtype=torch.float32)
bias = torch.tensor(1.5, dtype=torch.float32)

# We use matrix-vector multiplication
# PyTorch needs the weights to be a column vector for this, so we reshape it
scores = inputs_batch @ weights.reshape(-1, 1) + bias

print(f"Input Batch Shape: {inputs_batch.shape}")
print(f"Weights Shape: {weights.reshape(-1, 1).shape}")
print(f"---")
print(f"Scores for the batch:\n{scores}")
```
*Note: The rules for matrix multiplication are more complex than shown here. We'll explore this more when we build a full network layer.*

## 5. Practice Examples

### Practice 1: A Simple Neuron

You have a neuron with 2 inputs.
- `inputs = [2.0, 3.0]`
- `weights = [-0.5, 1.0]`
- `bias = -1.0`

Calculate the score `z` by hand and then verify your answer using PyTorch.

### Practice 2: The Effect of Bias

Using the same inputs and weights from Practice 1, what happens to the score if you change the bias to `+1.0`? What does this tell you about the role of the bias?

---

Now that we can calculate the neuron's internal score, we need to decide what to do with it. That's the job of the activation function, which we'll cover in the next lesson.

---

**Next Lesson**: [The Activation Function](03_the_activation_function.md)
