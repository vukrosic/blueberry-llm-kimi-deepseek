# Neural Network From Scratch: Implementing a Network

We've built a `Layer`. Now, we'll stack these layers to form a complete neural network. We will define our network as a class that inherits from PyTorch's `nn.Module`.

## 1. The Goal: Define the Network Architecture

We will build the network we designed in the first lesson:
- **Input Layer**: 2 features
- **Hidden Layer 1**: 16 neurons (ReLU activation)
- **Hidden Layer 2**: 8 neurons (ReLU activation)
- **Output Layer**: 1 neuron (Sigmoid activation)

This structure is defined by creating `Layer` instances inside our new `Network` class.

## 2. Designing the `Network` Class

A PyTorch `nn.Module` has two main parts:
1.  The `__init__` method, where you define all the layers the network will contain.
2.  The `forward` method, where you define how data flows *through* those layers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# We'll use PyTorch's built-in nn.Linear for convenience,
# as it's equivalent to the Layer class we designed.
class SimpleNetwork(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_outputs):
        super().__init__()

        # Define the layers
        self.hidden1 = nn.Linear(in_features=n_inputs, out_features=n_hidden1)
        self.hidden2 = nn.Linear(in_features=n_hidden1, out_features=n_hidden2)
        self.output = nn.Linear(in_features=n_hidden2, out_features=n_outputs)

    def forward(self, x):
        # This defines the data flow

        # Pass through first hidden layer, then apply ReLU
        x = self.hidden1(x)
        x = F.relu(x)

        # Pass through second hidden layer, then apply ReLU
        x = self.hidden2(x)
        x = F.relu(x)

        # Pass through output layer, then apply Sigmoid
        x = self.output(x)
        x = torch.sigmoid(x)

        return x
```

This is a complete definition of a neural network! `nn.Module` will automatically track all the parameters (weights and biases) from all the `nn.Linear` layers we defined.

## 3. The Forward Pass

Let's instantiate our network and pass a batch of data through it to see the `forward` method in action. This entire process of data flowing from input to output is called the **forward pass**.

```python
# --- Configuration ---
n_inputs = 2
n_hidden1 = 16
n_hidden2 = 8
n_outputs = 1
batch_size = 4

# --- Create the Network and Data ---
# Instantiate the network
model = SimpleNetwork(n_inputs, n_hidden1, n_hidden2, n_outputs)

# Create a batch of random input data
input_batch = torch.randn(batch_size, n_inputs)

print("--- Network Architecture ---")
print(model)

print("\n--- Forward Pass ---")
print(f"Input batch shape: {input_batch.shape}")

# Perform the forward pass
predictions = model.forward(input_batch)

print(f"Final predictions shape: {predictions.shape}")
print(f"Predictions:\n{predictions}")
```

The output `predictions` is a tensor of shape `(4, 1)`. It contains the output of the final Sigmoid neuron for each of the 4 samples in our input batch. These are the probabilities our (currently random and untrained) model is predicting.

## 4. `nn.Sequential`: A Simpler Way

For simple networks where the data just flows from one layer to the next, PyTorch provides a convenient container called `nn.Sequential` that builds the network for you.

```python
# Define the same model using nn.Sequential
sequential_model = nn.Sequential(
    nn.Linear(n_inputs, n_hidden1),
    nn.ReLU(),
    nn.Linear(n_hidden1, n_hidden2),
    nn.ReLU(),
    nn.Linear(n_hidden2, n_outputs),
    nn.Sigmoid()
)

print("\n--- Sequential Model ---")
print(sequential_model)

# It works exactly the same way
seq_predictions = sequential_model(input_batch)
print(f"\nSequential predictions shape: {seq_predictions.shape}")
```
 While `nn.Sequential` is handy, defining the `forward` method manually (as in our `SimpleNetwork` class) is more flexible and is required for more complex architectures with branching paths (like ResNets or Inception networks).

## 5. Practice Examples

### Practice 1: A Deeper Network
Define a new network class, `DeeperNetwork`, that has a third hidden layer with 4 neurons.

### Practice 2: A Regression Network
Define a network for a regression task. It should take 10 input features, have one hidden layer of 32 neurons, and produce a single continuous output. What activation function should you use on the output layer?

---

Our network can now make predictions. The next step is to calculate the loss for a whole batch of these predictions, which will prepare us for the training process.

---

**Next Lesson**: [The Chain Rule](04_the_chain_rule.md)

```
