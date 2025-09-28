# Activation Functions: ReLU

Welcome to the series on activation functions! Here we'll explore the most common activation functions used in modern neural networks, starting with the most popular of all: the **Rectified Linear Unit (ReLU)**.

## 1. The Goal: Understand ReLU

The ReLU function is defined as:
`f(z) = max(0, z)`

In simple terms:
- If the input `z` is positive, the output is just `z`.
- If the input `z` is negative, the output is `0`.

## 2. Visualizing ReLU

The function's graph clearly shows this behavior. It's a "hinge" at zero.

```python
import torch
import matplotlib.pyplot as plt

z = torch.linspace(-5, 5, 100)
output = torch.relu(z)

plt.figure(figsize=(8, 4))
plt.plot(z, output)
plt.title("ReLU Activation Function")
plt.xlabel("Input (z)")
plt.ylabel("Activation Output")
plt.grid(True)
plt.show()
```

## 3. Why is ReLU so Popular?

ReLU became the default choice for hidden layers in most neural networks for a few key reasons:

### a) It's Computationally Cheap
The `max(0, z)` operation is extremely fast for a computer to perform, much faster than the exponential calculations required for Sigmoid or Tanh. This means networks can train faster.

### b) It Mitigates the Vanishing Gradient Problem
In very deep networks, the gradients can become smaller and smaller as they are backpropagated through many layers, eventually "vanishing" to near-zero. This effectively stops the early layers from learning.

The derivative (gradient) of the ReLU function is:
- `1` for `z > 0`
- `0` for `z < 0`

Because the gradient is a constant `1` for positive inputs, it doesn't shrink as it passes through many ReLU layers. This allows gradients to flow more effectively through deep networks, enabling them to learn better.

## 4. The "Dying ReLU" Problem

ReLU is not perfect. It has a potential weakness called the "Dying ReLU" problem.

- If a neuron's weights are updated in such a way that its pre-activation score `z` is always negative for any input in the training data, that neuron will always output `0`.
- Since the gradient of ReLU is `0` for all negative inputs, no gradient will flow back through that neuron.
- The neuron's weights will never be updated again. It is effectively "dead" and will not participate in learning.

In practice, this is often not a major issue and can be mitigated by using a smaller learning rate or a variant of ReLU.

## 5. How to Use It

In PyTorch, you can use `torch.relu()` or the `torch.nn.ReLU` module.

```python
import torch

# A tensor of pre-activation scores
z = torch.tensor([-2.5, -0.1, 0.0, 1.5, 3.0])

# Apply ReLU
output = torch.relu(z)

print(f"Original scores: {z}")
print(f"After ReLU:      {output}")
```

## 6. Practice Examples

### Practice 1: Manual ReLU
Given the input tensor `z = torch.tensor([-10, -5, 0, 5, 10])`, calculate the output of the ReLU function by hand. Then, verify your result using `torch.relu()`.

### Practice 2: Leaky ReLU
A common variant to solve the "Dying ReLU" problem is **Leaky ReLU**. It allows a small, non-zero gradient when the input is negative.

`LeakyReLU(z) = max(0.01*z, z)`

Implement the Leaky ReLU function yourself in Python. Test it on the tensor from Practice 1. Then, compare your result to PyTorch's built-in `torch.nn.functional.leaky_relu()`.

---
ReLU is the workhorse of modern deep learning, but other functions are still vital, especially in the output layer. Next, we'll revisit the Sigmoid function in its modern context.
