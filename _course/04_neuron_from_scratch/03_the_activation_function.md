# Neuron From Scratch: The Activation Function

We've calculated the neuron's internal score (`z`). But a neuron doesn't just output this raw score. The final, crucial step is to pass this score through an **activation function**.

## 1. The Goal: Introduce Non-Linearity

Why can't we just use the raw score?

If we only used the linear step (`weights * inputs + bias`), our entire neural network, no matter how many layers, would just be one big linear function. A simple linear function can't learn the complex patterns found in real-world data like images, sound, and text.

**Activation functions introduce non-linearity**, allowing networks to learn much more complex relationships. They are the key to the power of deep learning.

## 2. What it Does

An activation function takes the raw score `z` and "squashes" it into a new value. It's a fixed function that decides the final output of the neuron based on the score.

`output = f(z)`

Where `f` is the activation function.

There are many different activation functions, each with different properties. Let's look at one of the most common ones: the **Sigmoid** function.

## 3. The Sigmoid Function

The Sigmoid function is defined as:
`σ(z) = 1 / (1 + e⁻ᶻ)`

It takes any real number `z` and squashes it to a value between 0 and 1.

- If `z` is a large positive number, `e⁻ᶻ` is close to 0, so `σ(z)` is close to 1.
- If `z` is a large negative number, `e⁻ᶻ` is very large, so `σ(z)` is close to 0.
- If `z` is 0, `σ(z)` is `1 / (1 + 1) = 0.5`.

This is perfect for binary classification problems, where the output can be interpreted as a probability (e.g., a 0.9 output means "90% probability of being true").

### Visualizing Sigmoid

Let's plot the Sigmoid function to see its characteristic "S" shape.

```python
import torch
import matplotlib.pyplot as plt

# Create a range of scores (z)
z = torch.linspace(-10, 10, 100)

# Apply the sigmoid function
output = torch.sigmoid(z)
# or manually: output = 1 / (1 + torch.exp(-z))

plt.figure(figsize=(8, 4))
plt.plot(z, output)
plt.title("Sigmoid Activation Function")
plt.xlabel("Score (z)")
plt.ylabel("Activation Output")
plt.grid(True)
plt.show()
```

## 4. How to Use It

Let's take the score we calculated in the previous lesson and apply the Sigmoid activation.

```python
import torch

# Our score from the last lesson
score = torch.tensor(2.22)

# Apply the sigmoid activation function
activation = torch.sigmoid(score)

print(f"Original score: {score:.4f}")
print(f"After Sigmoid activation: {activation:.4f}")
```
The output (around 0.9) is the final output of our neuron. It's a high value, so the neuron is "firing" strongly.

## 5. Practice Examples

### Practice 1: Test the Limits

What is the Sigmoid activation for the following scores?
- `z = 100`
- `z = -100`
- `z = 0`
Does this match what you expect from the graph?

### Practice 2: Another Activation Function - ReLU

The Rectified Linear Unit (ReLU) is another extremely popular activation function. It's even simpler than Sigmoid:
`ReLU(z) = max(0, z)`

It simply returns `z` if `z` is positive, and `0` otherwise.

Calculate the ReLU activation for these scores:
- `z = 5.5`
- `z = -3.2`
- `z = 0`

PyTorch has a function for this: `torch.relu()`. Verify your answers with code.

---
We now have all the conceptual pieces of a neuron! In the next lesson, we'll put them all together and build a neuron as a reusable piece of code.
