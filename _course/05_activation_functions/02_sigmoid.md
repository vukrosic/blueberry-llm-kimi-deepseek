# Activation Functions: Sigmoid

The Sigmoid function was once the most popular activation function, especially in the early days of deep learning. While it has been largely replaced by ReLU in hidden layers, it still has a crucial role to play.

## 1. The Goal: Understand Sigmoid

The Sigmoid function is defined as:
`σ(z) = 1 / (1 + e⁻ᶻ)`

It takes any real number and squashes it to a value between 0 and 1.

## 2. Visualizing Sigmoid

The function has a characteristic "S" shape.

```python
import torch
import matplotlib.pyplot as plt

z = torch.linspace(-10, 10, 100)
output = torch.sigmoid(z)

plt.figure(figsize=(8, 4))
plt.plot(z, output)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input (z)")
plt.ylabel("Activation Output")
plt.grid(True)
plt.show()
```

## 3. Modern Usage: Output Layers

Today, Sigmoid is almost exclusively used in the **output layer** of a neural network for **binary classification** problems.

Because it squashes any value into the `(0, 1)` range, its output can be interpreted as a **probability**.

- **Example**: A model that predicts if an email is spam or not.
    - An output of `0.95` means the model is 95% confident the email is spam.
    - An output of `0.02` means the model is 98% confident the email is *not* spam.

When paired with the Binary Cross-Entropy (BCE) loss function, it provides a powerful mechanism for training binary classifiers.

## 4. Why Not in Hidden Layers?

Sigmoid fell out of favor for hidden layers for two main reasons, both related to its gradients:

### a) The Vanishing Gradient Problem
The gradient of the Sigmoid function is only significant for inputs between roughly -3 and 3. For inputs outside this range, the gradient is extremely close to zero.

This means that during backpropagation, if a neuron's pre-activation score `z` is very large or very small, almost no gradient will flow back through it. In a deep network, these tiny gradients are multiplied together across many layers, quickly "vanishing" to zero. This stops the network from learning effectively.

### b) Not Zero-Centered
The output of the Sigmoid function is always positive (between 0 and 1). This means that all the gradients flowing into a layer from a Sigmoid-activated layer will have the same sign. This can lead to inefficient, zig-zagging convergence during gradient descent. While this is a more subtle issue, it contributes to slower training compared to zero-centered activation functions like Tanh.

## 5. How to Use It

Using Sigmoid in PyTorch is straightforward.

```python
import torch

# A tensor of pre-activation scores (logits) from the output layer
z = torch.tensor([-1.5, 0.0, 2.5, 5.0])

# Apply Sigmoid to get probabilities
probabilities = torch.sigmoid(z)

print(f"Original scores: {z}")
print(f"Probabilities:   {probabilities}")
```

## 6. Practice Examples

### Practice 1: Probability Threshold
Using the `probabilities` tensor from the example above, write code to convert them into final predictions (0 or 1) using a threshold of 0.5.

### Practice 2: The Gradient at the Extremes
The derivative of the sigmoid function is `σ(z) * (1 - σ(z))`.
Calculate the gradient for `z = 10` and `z = 0`.
What does this tell you about the vanishing gradient problem?

---
Next, we'll look at Tanh, a close cousin of Sigmoid that addresses one of its key problems.

---

**Next Lesson**: [Tanh](03_tanh.md)
