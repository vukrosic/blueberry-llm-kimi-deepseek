# Neuron From Scratch: The Concept of Learning

We've reached the final and most magical step: **learning**.

We know our neuron's prediction is wrong, and we've measured how wrong it is using a loss function. We also know we need to go "downhill" on the loss landscape to find better weights. The process of actually taking that step downhill is called **gradient descent**.

## 1. The Goal: Update Weights to Reduce Loss

Learning is simply the process of adjusting the neuron's weights and bias in the direction that reduces the loss.

The core formula for this update is:
`new_weight = old_weight - learning_rate * gradient`

Let's break this down.

## 2. The Gradient: Which Way is Downhill?

As we discussed, the gradient of the loss with respect to a weight tells us how the loss will change if we increase that weight. It's the slope of the loss landscape at our current position.

- If `gradient > 0`: Increasing the weight increases the loss. To decrease the loss, we need to *decrease* the weight.
- If `gradient < 0`: Increasing the weight decreases the loss. To decrease the loss, we need to *increase* the weight.

Notice that in both cases, we want to move in the *opposite* direction of the gradient. This is why the algorithm is called gradient **descent**.

Calculating this gradient is the job of **backpropagation**, which is a topic for a later series. For now, just know that PyTorch can do this for us automatically with a magical function: `loss.backward()`.

## 3. The Learning Rate: How Big of a Step to Take?

The **learning rate** (often denoted as `α` or `lr`) is a small number (e.g., 0.01) that controls how big of a step we take downhill.

- **Too large a learning rate**: We might overshoot the bottom of the valley and end up on the other side, with an even higher loss. It's like trying to walk down a hill by taking giant leaps.
- **Too small a learning rate**: We will definitely make progress downhill, but it will take a very, very long time to get to the bottom. It's like shuffling your feet one centimeter at a time.

Choosing a good learning rate is a critical part of training a neural network.

## 4. The Full Process (The Training Loop)

Let's put all the concepts from this series together into a single "training step".

```python
import torch

# --- Setup ---
# Let's use our Neuron class again
class Neuron:
    def __init__(self, num_inputs):
        # We need to tell PyTorch to track gradients for these tensors
        self.weights = torch.randn(num_inputs, requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)

    def forward(self, inputs):
        score = torch.dot(self.weights, inputs) + self.bias
        return torch.sigmoid(score)

# --- The Training Step ---

# 1. Initialize our neuron, data, and the truth
neuron = Neuron(2)
inputs = torch.tensor([1.0, 1.0]) # A student who studied and slept a lot
truth = torch.tensor(1.0)       # They actually passed

# Define our learning rate and loss function
learning_rate = 0.1
loss_function = torch.nn.BCELoss()

print(f"Initial weights: {neuron.weights.data}")
print(f"Initial bias: {neuron.bias.data}")

# 2. FORWARD PASS: Make a prediction
prediction = neuron.forward(inputs)

# 3. CALCULATE LOSS: See how wrong we are
loss = loss_function(prediction, truth)

print(f"\nPrediction: {prediction.item():.4f}, Loss: {loss.item():.4f}")

# 4. BACKWARD PASS: Calculate gradients
# This is the magic step! PyTorch calculates the gradient of the loss
# with respect to every tensor that has `requires_grad=True`.
loss.backward()

# The calculated gradients are stored in the .grad attribute
print(f"Gradient for weights: {neuron.weights.grad}")
print(f"Gradient for bias: {neuron.bias.grad}")

# 5. UPDATE WEIGHTS: Take a step downhill
# We use `torch.no_grad()` to tell PyTorch not to track this operation,
# as it's part of the optimization process itself.
with torch.no_grad():
    neuron.weights -= learning_rate * neuron.weights.grad
    neuron.bias -= learning_rate * neuron.bias.grad

    # We must zero out the gradients after using them,
    # or they will accumulate on the next pass.
    neuron.weights.grad.zero_()
    neuron.bias.grad.zero_()

print(f"\n--- After 1 Step ---")
print(f"New weights: {neuron.weights.data}")
print(f"New bias: {neuron.bias.data}")

# Let's see if the loss improved
new_prediction = neuron.forward(inputs)
new_loss = loss_function(new_prediction, truth)
print(f"New Prediction: {new_prediction.item():.4f}, New Loss: {new_loss.item():.4f}")
```
If you run this, you'll see that the `New Loss` is smaller than the initial loss. We have successfully taught the neuron a tiny little bit! A full training process just repeats this loop thousands of times with thousands of data points.

## 5. Series Recap

Congratulations! You've seen all the core components of how a single neuron learns:
1.  **What a Neuron Is**: A unit that weighs inputs.
2.  **The Linear Step**: `z = w · x + b`
3.  **The Activation Function**: `output = f(z)`
4.  **Making a Prediction**: The `forward` pass.
5.  **Calculating Loss**: Measuring the error.
6.  **Learning**: Using the gradient of the loss to update the weights.

You now have the conceptual foundation to understand entire neural networks.

---
Next, we'll dive deeper into one of the key components: the activation function.

---

**Next Lesson**: [ReLU](../05_activation_functions/01_relu.md) (Activation Functions Module)
