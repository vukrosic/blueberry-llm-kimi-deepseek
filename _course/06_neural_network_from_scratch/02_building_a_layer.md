# Neural Network From Scratch: Building a Layer

A layer is just a collection of neurons that process inputs together. When we move from a single neuron to a full layer, we must switch from vector operations to **matrix operations**.

## 1. The Goal: Create a `Layer` Class

We will build a `Layer` class that represents a fully-connected (or dense) layer. It will:
1.  Be initialized with the number of input features it expects (`n_inputs`) and the number of neurons it contains (`n_neurons`).
2.  Contain a **weight matrix** and a **bias vector**.
3.  Have a `forward` method that performs the layer's calculation for a batch of data.

## 2. From a Neuron to a Layer

Recall the calculation for a single neuron:
`z = (w₁x₁ + w₂x₂ + ...) + b` (a dot product)

Now, imagine we have `n_neurons` in our layer. Each neuron has its own set of weights. We can organize these weights into a **matrix**.

- **Input `x`**: A vector of size `n_inputs`.
- **Weight Matrix `W`**: A matrix of shape `(n_inputs, n_neurons)`. Each *column* of this matrix is the weight vector for one neuron.
- **Bias Vector `b`**: A vector of size `n_neurons`. Each element is the bias for one neuron.

The linear step for the entire layer becomes a **matrix-vector multiplication**:
`z = x @ W + b`

- `x` (1 x `n_inputs`) @ `W` (`n_inputs` x `n_neurons`) -> (1 x `n_neurons`)
- `+ b` (1 x `n_neurons`) -> (1 x `n_neurons`)

The result `z` is a vector containing the score for each neuron in the layer.

## 3. Designing the `Layer` Class

Let's build this using PyTorch's `nn.Module`, which is the base class for all neural network modules. Using `nn.Module` helps PyTorch automatically track our layer's parameters (`W` and `b`).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer(nn.Module):
    def __init__(self, n_inputs, n_neurons, activation=None):
        super().__init__() # Essential for all nn.Module classes

        # Create the weight matrix and bias vector.
        # By wrapping them in nn.Parameter, we tell PyTorch that these
        # are learnable parameters of the model.
        self.weights = nn.Parameter(torch.randn(n_inputs, n_neurons))
        self.biases = nn.Parameter(torch.zeros(n_neurons))

        # Store the activation function
        self.activation = activation

    def forward(self, x):
        # The linear step: x @ W + b
        score = x @ self.weights + self.biases

        # The activation step
        if self.activation:
            return self.activation(score)
        else:
            return score
```

This is the fundamental structure of a `Linear` (or `Dense`) layer in any deep learning framework!

## 4. Using Our `Layer`

Let's create a layer that takes 5 input features and has 3 neurons. We'll pass a batch of 2 inputs through it.

```python
# Configuration
n_inputs = 5
n_neurons = 3
batch_size = 2

# Create a layer with a ReLU activation
my_layer = Layer(n_inputs, n_neurons, activation=F.relu)

# Let's inspect its parameters (weights and biases)
# PyTorch automatically tracks them
for name, param in my_layer.named_parameters():
    print(f"Parameter '{name}' shape: {param.shape}")

# Create a batch of random input data
input_batch = torch.randn(batch_size, n_inputs)

print(f"\nInput batch shape: {input_batch.shape}")

# Perform the forward pass
output = my_layer.forward(input_batch)

print(f"Layer output shape: {output.shape}")
print(f"Layer output:\n{output}")
```
The input has shape `(2, 5)`. The output has shape `(2, 3)`. We passed 2 samples through the layer, and for each sample, we got an output value from each of the 3 neurons. This is exactly what we expect.

## 5. Practice Examples

### Practice 1: An Output Layer
Create a `Layer` instance that would be suitable as the output layer for a binary classification problem. It should take 16 inputs (from a previous hidden layer) and have a Sigmoid activation function. What should `n_neurons` be?

### Practice 2: No Activation
Create a `Layer` instance that would be suitable as the output layer for a regression problem (like predicting a house price). It should take 8 inputs. What should `n_neurons` and `activation` be?

### Practice 3: PyTorch's Built-in Layer
PyTorch already has a powerful, optimized class for this: `nn.Linear`. It works almost identically to our `Layer` class.

```python
# Our layer
layer1 = Layer(n_inputs=10, n_neurons=5)

# PyTorch's built-in equivalent
linear_layer = nn.Linear(in_features=10, out_features=5)

# Notice the parameters are named the same!
print("--- PyTorch nn.Linear Layer ---")
for name, param in linear_layer.named_parameters():
    print(f"Parameter '{name}' shape: {param.shape}")
```
Modify the code to pass a batch of data through the `nn.Linear` layer.

---
We can now build layers. In the next lesson, we'll stack these layers together to create the full architecture of our neural network.
