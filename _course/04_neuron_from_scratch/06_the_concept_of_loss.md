# Neuron From Scratch: The Concept of Loss

Our neuron made a prediction, and it was wrong. In machine learning, we don't just say "wrong"; we quantify *how* wrong it was. This measurement of error is called **loss**.

## 1. The Goal: Measure the Neuron's Error

A **loss function** (or cost function) is a function that takes the neuron's prediction and the ground truth label and outputs a single number: the loss.

- If the prediction is very close to the truth, the loss will be small.
- If the prediction is far from the truth, the loss will be large.

The goal of training is to **minimize the loss**.

## 2. A Simple Loss Function: Mean Squared Error (MSE)

One of the most common loss functions is Mean Squared Error (MSE). While it's more common for regression (predicting a continuous value), it's the easiest to understand for our first example.

The formula for the squared error for a single prediction is:
`Error = (prediction - truth)²`

We square the difference to ensure the error is always positive and to penalize larger errors more heavily.

## 3. Calculating the Loss

Let's go back to our `pass_fail_neuron`.
- **Inputs**: `[1.0, 0.8]`
- **Prediction (output probability)**: `0.19`
- **Ground Truth**: `1` (the student actually passed)

Let's calculate the squared error.

```python
import torch

# The neuron's output
prediction = torch.tensor(0.19)

# The actual, correct answer
truth = torch.tensor(1.0)

# Calculate the squared error
loss = (prediction - truth)**2

print(f"Prediction: {prediction:.2f}")
print(f"Truth: {truth:.2f}")
print(f"---")
print(f"Loss (Squared Error): {loss.item():.4f}")
```
The result is `(0.19 - 1.0)² = (-0.81)² ≈ 0.6561`. This is a relatively high number, indicating our neuron was quite wrong.

## 4. The Landscape of Loss

Now, here is the most important concept in deep learning.

The **loss is a function of the neuron's weights and bias**.

If we were to change the weights and bias, the neuron's prediction would change, and therefore the loss would change.

Imagine a 3D landscape where:
- The `x` and `y` axes represent the values of our two weights.
- The `z` axis (the height) represents the loss for those weights.

Our goal is to find the combination of weights that corresponds to the **lowest point in this landscape**. This lowest point is the **minimum loss**.

<img src="https://i.imgur.com/tW5v2sD.png" width="500">
*(Image: A typical loss landscape. We are at a high point and want to get to the valley.)*

## 5. How Do We Find the Bottom?

We are currently at some random point on this landscape, and our loss is high. We need to figure out which direction to move in to go downhill.

The answer is **gradient descent**.

The **gradient** is a vector that points in the direction of the steepest ascent (uphill). Therefore, the **negative gradient** points in the direction of the steepest descent (downhill).

By calculating the gradient of the loss with respect to the weights and bias, we can find the direction to "nudge" our weights and bias to reduce the loss.

## 6. Practice Examples

### Practice 1: A Better Prediction

Let's say we tweaked our neuron's weights, and now for the same student, it predicts `0.85`. The ground truth is still `1`.
Calculate the new squared error. Is it lower or higher than before?

### Practice 2: A Perfect Prediction

What would the loss be if the neuron predicted exactly `1.0` and the ground truth was `1.0`?

### Practice 3: Binary Cross-Entropy

For classification tasks, Mean Squared Error isn't the best choice. A much more common loss function is **Binary Cross-Entropy (BCE)**.

The formula is more complex:
`Loss = -[ y * log(p) + (1-y) * log(1-p) ]`
Where `y` is the truth (0 or 1) and `p` is the predicted probability.

PyTorch has a function for this: `torch.nn.BCELoss`.

```python
import torch

loss_fn = torch.nn.BCELoss()

prediction = torch.tensor(0.19)
truth = torch.tensor(1.0)

loss = loss_fn(prediction, truth)
print(f"BCE Loss: {loss.item():.4f}")
```
Calculate the BCE loss for the prediction of `0.85`. Is the trend the same as with MSE (does a better prediction give a lower loss)?

---
We know we need to go "downhill" on the loss landscape. In the final lesson of this series, we'll explore the concept of how we actually take that step.
