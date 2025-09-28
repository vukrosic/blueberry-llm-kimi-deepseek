
# Lesson 7: Implementing Backpropagation

It's time to translate the theory of backpropagation into code. We'll add a `backward` method to our neural network to calculate the gradients, and an `update` method to adjust the weights.

Let's assume we have a `NeuralNetwork` class with layers, as we built in a previous lesson.

## The `backward` Method

The `backward` method will take the gradient of the loss with respect to the network's output (`dL/da_out`) and propagate it backward.

```python
import numpy as np

# Let's assume we have a Layer class
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
        self.input = None
        self.z = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.weights, self.input) + self.biases
        # For simplicity, let's use a linear activation (f(z) = z)
        # so the derivative f'(z) is 1
        return self.z

    def backward(self, d_output):
        # Gradient of the loss with respect to the weights
        self.d_weights = np.dot(d_output, self.input.T)

        # Gradient of the loss with respect to the biases
        self.d_biases = np.sum(d_output, axis=1, keepdims=True)

        # Gradient of the loss with respect to the input
        # This is what gets passed back to the previous layer
        d_input = np.dot(self.weights.T, d_output)
        return d_input

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_loss):
        # Start from the last layer and go backwards
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.weights -= learning_rate * layer.d_weights
            layer.biases -= learning_rate * layer.d_biases

```

*Note: For simplicity, the code above uses a linear activation function where `f'(z) = 1`. For other activation functions like sigmoid or ReLU, you would need to multiply by their derivatives during the backward pass.*

## The Training Loop

Now, let's see how these methods fit into a training loop.

```python
# 1. Initialize the network
nn = NeuralNetwork()
nn.add_layer(Layer(input_size=2, output_size=3))
nn.add_layer(Layer(input_size=3, output_size=1))

# 2. Define loss function and its derivative
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

# 3. The training loop
for epoch in range(1000):
    # a. Forward pass
    predictions = nn.forward(x_train)

    # b. Calculate loss
    loss = mse_loss(predictions, y_train)

    # c. Backward pass
    d_loss = mse_loss_derivative(predictions, y_train)
    nn.backward(d_loss)

    # d. Update weights
    nn.update(learning_rate=0.01)
```

### What's Happening?

1.  We make a prediction with the **forward pass**.
2.  We calculate how wrong the prediction is using the **loss function**.
3.  We use the derivative of the loss function to start the **backward pass**. The `backward` method calculates the gradients for all the weights and biases in the network.
4.  The `update` method takes a small step in the direction opposite to the gradient, nudging the weights towards values that will reduce the loss.

## Conclusion

You now have a conceptual and practical understanding of how a neural network learns! From the chain rule to a full implementation, you've seen the entire process of backpropagation.

This is the foundation of how modern neural networks are trained. While deep learning libraries like PyTorch and TensorFlow automate this for you, understanding what happens under the hood is incredibly valuable.
