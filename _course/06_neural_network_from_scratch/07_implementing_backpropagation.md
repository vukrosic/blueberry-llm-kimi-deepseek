# Lesson 7: Implementing Backpropagation Step-by-Step

It's time to translate the theory of backpropagation into code. We'll build up our `NeuralNetwork` class piece by piece, starting with a single layer and progressively adding the components needed for learning.

*Note: For simplicity, our examples will use a linear activation function where `f(z) = z`. This means the derivative `f'(z)` is 1, which keeps the focus on the mechanics of backpropagation.*

---

### Part 1: The Layer's Forward Pass

First, let's define a single `Layer`. It needs to initialize its weights and biases and be able to perform a forward pass.

```python
import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases randomly
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
        self.input = None

    def forward(self, input_data):
        # Store the input for use in the backward pass
        self.input = input_data
        # Perform the linear step
        return np.dot(self.weights, self.input) + self.biases

# --- Let's test it ---
# Create a layer that takes 2 inputs and has 3 neurons
layer1 = Layer(input_size=2, output_size=3)

# Create some dummy input data
# (2 features, 1 sample)
input_sample = np.array([[0.5], [0.2]])

# Perform a forward pass
output = layer1.forward(input_sample)
print("Output of the layer:")
print(output)
```

In this step, we've created a layer that can take data and produce an output. But it can't learn yet.

---

### Part 2: The Layer's Backward Pass

Now, let's add the `backward` method. This method calculates the gradients—how much the loss changes with respect to the layer's weights and biases.

```python
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
        self.input = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, d_output):
        # Gradient of the loss w.r.t. weights
        # dL/dW = dL/d_output * d_output/dW = d_output * input
        self.d_weights = np.dot(d_output, self.input.T)

        # Gradient of the loss w.r.t. biases
        # dL/db = dL/d_output * d_output/db = d_output * 1
        self.d_biases = np.sum(d_output, axis=1, keepdims=True)

        # Gradient of the loss w.r.t. input
        # This is what gets passed back to the previous layer
        # dL/d_input = dL/d_output * d_output/d_input = W.T * d_output
        d_input = np.dot(self.weights.T, d_output)
        return d_input

# --- Let's test it ---
layer2 = Layer(input_size=3, output_size=1)
# Let's imagine this layer received some data and produced an output
layer2.input = np.array([[0.5], [0.2], [0.8]]) 

# And let's assume the gradient coming from the loss function is 0.4
d_loss_output = np.array([[0.4]])

# Now, let's do the backward pass
d_loss_input = layer2.backward(d_loss_output)

print("Gradient w.r.t. weights (d_weights):")
print(layer2.d_weights)
print("\nGradient w.r.t. input (to be passed back):")
print(d_loss_input)
```
The `backward` method gives us the gradients (`d_weights` and `d_biases`), which tell us how to update our parameters.

---

### Part 3: Assembling the Full Network

Now we can create a `NeuralNetwork` class to manage the layers.

```python
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        # Pass input through each layer sequentially
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_loss):
        # Pass gradients backward through each layer
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss)

# --- Let's test it ---
nn = NeuralNetwork()
nn.add_layer(Layer(input_size=2, output_size=3))
nn.add_layer(Layer(input_size=3, output_size=1))

# Let's do a full forward and backward pass
input_data = np.array([[0.5], [0.2]])
output = nn.forward(input_data)
print("Network output:", output)

# Assume the gradient of the loss is 0.4
d_loss = np.array([[0.4]])
nn.backward(d_loss)

# We can inspect the gradients in the first layer
print("\nGradients for the first layer's weights:")
print(nn.layers[0].d_weights)
```

---

### Part 4: Updating the Weights

We have the gradients, so now we need a way to use them to update the weights. This is done with an `update` method and a `learning_rate`.

```python
class Layer:
    # ... (previous methods from Part 2) ...
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
        self.input = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, d_output):
        self.d_weights = np.dot(d_output, self.input.T)
        self.d_biases = np.sum(d_output, axis=1, keepdims=True)
        d_input = np.dot(self.weights.T, d_output)
        return d_input
        
    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

class NeuralNetwork:
    # ... (previous methods from Part 3) ...
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_loss):
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss)
            
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
```
The update rule `self.weights -= learning_rate * self.d_weights` is the core of gradient descent.

---

### Part 5: The Complete Training Loop

Now we can put all the pieces together into a final, runnable example.

```python
import numpy as np

# (Paste the full Layer and NeuralNetwork classes from Part 4 here)

# --- Define Loss Function ---
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

# --- 1. Create the Network ---
nn = NeuralNetwork()
nn.add_layer(Layer(input_size=2, output_size=3))
nn.add_layer(Layer(input_size=3, output_size=1))

# --- 2. Create Training Data ---
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]]).reshape(4, 2, 1)
y_train = np.array([[[0]], [[1]], [[1]], [[0]]]).reshape(4, 1, 1)

# --- 3. The Training Loop ---
epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    total_loss = 0
    for x, y in zip(x_train, y_train):
        # a. Forward pass
        prediction = nn.forward(x)

        # b. Calculate loss
        total_loss += mse_loss(prediction, y)

        # c. Backward pass (get gradients)
        d_loss = mse_loss_derivative(prediction, y)
        nn.backward(d_loss)

        # d. Update weights
        nn.update(learning_rate)
    
    # Print average loss for the epoch
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(x_train)}")

print("\n--- Final Predictions ---")
for x, y in zip(x_train, y_train):
    prediction = nn.forward(x)
    print(f"Input: {x.ravel()}, Prediction: {prediction.ravel()}, Actual: {y.ravel()}")
```
This complete example shows how the forward pass, backward pass, and update step all work together over many epochs to train the network. You can run this code and watch the loss decrease as the network learns!