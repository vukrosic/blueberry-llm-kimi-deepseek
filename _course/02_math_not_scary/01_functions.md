# Functions: The Foundation of Neural Networks

## Table of Contents
1. [What are Functions?](#what-are-functions)
2. [Types of Functions](#types-of-functions)
3. [Function Composition](#function-composition)
4. [Functions in Neural Networks](#functions-in-neural-networks)
5. [Activation Functions](#activation-functions)
6. [Loss Functions](#loss-functions)
7. [Practical Examples](#practical-examples)

## What are Functions?

A **function** is a mathematical relationship that maps inputs to outputs. In simple terms, it's like a machine that takes something in and gives something back out.

### Mathematical Definition
A function f: A → B maps every element in set A to exactly one element in set B.

### Notation
- f(x) = y (read as "f of x equals y")
- x is the input (independent variable)
- y is the output (dependent variable)

### Simple Examples

#### Hand Calculation Examples

**Example 1: Linear Function f(x) = 2x + 3**

Let's calculate f(x) for different values step by step:

Don't confuse f(1) and 2(1). f(1) means passing 1 into function f, and 2(1) mean 2*1.

For x = 1:
- f(1) = 2(1) + 3 = 2 + 3 = 5

For x = 0:
- f(0) = 2(0) + 3 = 0 + 3 = 3

For x = -1:
- f(-1) = 2(-1) + 3 = -2 + 3 = 1

**Example 2: Quadratic Function f(x) = x² + 2x + 1**

Let's calculate f(x) for different values step by step:

For x = 2:
- f(2) = (2)² + 2(2) + 1 = 4 + 4 + 1 = 9

For x = 0:
- f(0) = (0)² + 2(0) + 1 = 0 + 0 + 1 = 1

For x = -1:
- f(-1) = (-1)² + 2(-1) + 1 = 1 - 2 + 1 = 0

#### Code Examples

```python
# Linear function: f(x) = 2x + 3
def linear_function(x):
    return 2 * x + 3

# Test the function
print(f"f(1) = {linear_function(1)}")  # Output: f(1) = 5
print(f"f(0) = {linear_function(0)}")  # Output: f(0) = 3
print(f"f(-1) = {linear_function(-1)}")  # Output: f(-1) = 1
```

```python
# Quadratic function: f(x) = x² + 2x + 1
def quadratic_function(x):
    return x**2 + 2*x + 1

# Test the function
print(f"f(2) = {quadratic_function(2)}")  # Output: f(2) = 9
print(f"f(0) = {quadratic_function(0)}")  # Output: f(0) = 1
print(f"f(-1) = {quadratic_function(-1)}")  # Output: f(-1) = 0
```

## Types of Functions

### 1. Linear Functions
Linear functions have the form: f(x) = mx + b

Where:
- m is the slope (how steep the line is)
- b is the y-intercept (where the line crosses the y-axis)

#### Hand Calculation Examples

**Example: f(x) = 2x + 1**

Let's create a table of values:

| x | f(x) = 2x + 1 | Calculation |
|---|---------------|-------------|
| -2 | f(-2) = 2(-2) + 1 = -4 + 1 = -3 | 2(-2) + 1 = -3 |
| -1 | f(-1) = 2(-1) + 1 = -2 + 1 = -1 | 2(-1) + 1 = -1 |
| 0 | f(0) = 2(0) + 1 = 0 + 1 = 1 | 2(0) + 1 = 1 |
| 1 | f(1) = 2(1) + 1 = 2 + 1 = 3 | 2(1) + 1 = 3 |
| 2 | f(2) = 2(2) + 1 = 4 + 1 = 5 | 2(2) + 1 = 5 |

**Example: f(x) = -0.5x + 3**

Let's create a table of values:

| x | f(x) = -0.5x + 3 | Calculation |
|---|------------------|-------------|
| -2 | f(-2) = -0.5(-2) + 3 = 1 + 3 = 4 | -0.5(-2) + 3 = 4 |
| -1 | f(-1) = -0.5(-1) + 3 = 0.5 + 3 = 3.5 | -0.5(-1) + 3 = 3.5 |
| 0 | f(0) = -0.5(0) + 3 = 0 + 3 = 3 | -0.5(0) + 3 = 3 |
| 1 | f(1) = -0.5(1) + 3 = -0.5 + 3 = 2.5 | -0.5(1) + 3 = 2.5 |
| 2 | f(2) = -0.5(2) + 3 = -1 + 3 = 2 | -0.5(2) + 3 = 2 |

```python
import numpy as np
import matplotlib.pyplot as plt

# Linear function examples
def linear_1(x):
    return 2*x + 1

def linear_2(x):
    return -0.5*x + 3

# Plot linear functions
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, linear_1(x), label='f(x) = 2x + 1', linewidth=2)
plt.plot(x, linear_2(x), label='f(x) = -0.5x + 3', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Linear Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Polynomial Functions
Functions with powers of x: f(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀

#### Hand Calculation Examples

**Example: f(x) = x³ - 3x² + 2x + 1**

Let's calculate f(x) for different values step by step:

For x = 1:
- f(1) = (1)³ - 3(1)² + 2(1) + 1
- f(1) = 1 - 3(1) + 2 + 1
- f(1) = 1 - 3 + 2 + 1
- f(1) = 1

For x = 2:
- f(2) = (2)³ - 3(2)² + 2(2) + 1
- f(2) = 8 - 3(4) + 4 + 1
- f(2) = 8 - 12 + 4 + 1
- f(2) = 1

For x = 0:
- f(0) = (0)³ - 3(0)² + 2(0) + 1
- f(0) = 0 - 0 + 0 + 1
- f(0) = 1

**Example: f(x) = x⁴ - 4x² + 3**

Let's calculate f(x) for different values step by step:

For x = 1:
- f(1) = (1)⁴ - 4(1)² + 3
- f(1) = 1 - 4(1) + 3
- f(1) = 1 - 4 + 3
- f(1) = 0

For x = 2:
- f(2) = (2)⁴ - 4(2)² + 3
- f(2) = 16 - 4(4) + 3
- f(2) = 16 - 16 + 3
- f(2) = 3

For x = 0:
- f(0) = (0)⁴ - 4(0)² + 3
- f(0) = 0 - 0 + 3
- f(0) = 3

```python
# Polynomial function examples
def cubic_function(x):
    return x**3 - 3*x**2 + 2*x + 1

def quartic_function(x):
    return x**4 - 4*x**2 + 3

# Plot polynomial functions
x = np.linspace(-3, 3, 100)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, cubic_function(x), label='f(x) = x³ - 3x² + 2x + 1', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Cubic Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, quartic_function(x), label='f(x) = x⁴ - 4x² + 3', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Quartic Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3. Exponential Functions
Functions with constant base raised to variable power: f(x) = aˣ

```python
# Exponential function examples
def exponential_function(x):
    return 2**x

def exponential_e(x):
    return np.exp(x)

# Plot exponential functions
x = np.linspace(-2, 3, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, exponential_function(x), label='f(x) = 2ˣ', linewidth=2)
plt.plot(x, exponential_e(x), label='f(x) = eˣ', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Exponential Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to better visualize exponential growth
plt.show()
```

### 4. Trigonometric Functions
Functions based on angles and periodic behavior

```python
# Trigonometric function examples
def sine_function(x):
    return np.sin(x)

def cosine_function(x):
    return np.cos(x)

# Plot trigonometric functions
x = np.linspace(-2*np.pi, 2*np.pi, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, sine_function(x), label='f(x) = sin(x)', linewidth=2)
plt.plot(x, cosine_function(x), label='f(x) = cos(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()
```

## Function Composition

Function composition is combining two or more functions where the output of one becomes the input of another.

### Mathematical Notation
(f ∘ g)(x) = f(g(x))

### Examples

#### Hand Calculation Examples

**Example: f(x) = 2x - 3 and g(x) = x² + 1**

Let's find (f ∘ g)(x) = f(g(x)) step by step:

First, let's understand what g(x) does:
- g(x) = x² + 1

Now, (f ∘ g)(x) = f(g(x)) = f(x² + 1)

Since f(x) = 2x - 3, we substitute (x² + 1) for x:
- f(x² + 1) = 2(x² + 1) - 3
- f(x² + 1) = 2x² + 2 - 3
- f(x² + 1) = 2x² - 1

So (f ∘ g)(x) = 2x² - 1

Let's verify with specific values:

For x = 0:
- g(0) = (0)² + 1 = 0 + 1 = 1
- f(g(0)) = f(1) = 2(1) - 3 = 2 - 3 = -1
- (f ∘ g)(0) = 2(0)² - 1 = 0 - 1 = -1 ✓

For x = 1:
- g(1) = (1)² + 1 = 1 + 1 = 2
- f(g(1)) = f(2) = 2(2) - 3 = 4 - 3 = 1
- (f ∘ g)(1) = 2(1)² - 1 = 2 - 1 = 1 ✓

For x = 2:
- g(2) = (2)² + 1 = 4 + 1 = 5
- f(g(2)) = f(5) = 2(5) - 3 = 10 - 3 = 7
- (f ∘ g)(2) = 2(2)² - 1 = 8 - 1 = 7 ✓

```python
# Define two functions
def g(x):
    return x**2 + 1

def f(x):
    return 2*x - 3

# Composition: (f ∘ g)(x) = f(g(x))
def composition_fg(x):
    return f(g(x))

# Test composition
x_values = [0, 1, 2, 3]
for x in x_values:
    g_result = g(x)
    fg_result = composition_fg(x)
    print(f"x={x}: g(x)={g_result}, (f∘g)(x)={fg_result}")
```

### Neural Network Analogy
In neural networks, function composition is fundamental:

```python
# Simplified neural network layer
def neural_layer(input_data, weights, bias, activation):
    """
    A neural network layer is a composition of:
    1. Linear transformation: weights @ input + bias
    2. Activation function: activation(linear_output)
    """
    linear_output = np.dot(weights, input_data) + bias
    return activation(linear_output)

# Example usage
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Create a simple layer
input_data = np.array([1, 2, 3])
weights = np.array([[0.5, -0.3, 0.8],
                    [-0.2, 0.7, -0.1]])
bias = np.array([0.1, -0.2])

# Apply the layer with ReLU activation
output = neural_layer(input_data, weights, bias, relu)
print(f"Neural layer output: {output}")
```

## Functions in Neural Networks

### 1. Forward Propagation
Forward propagation is essentially function composition through multiple layers.

```python
class SimpleNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        """Forward propagation through the network"""
        a = x.reshape(-1, 1)  # Reshape to column vector
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = self.relu(z)  # Apply activation function
        
        return a.flatten()
    
    def relu(self, x):
        return np.maximum(0, x)

# Create and test a simple network
network = SimpleNeuralNetwork([3, 4, 2])
input_data = np.array([1, 2, 3])
output = network.forward(input_data)
print(f"Network input: {input_data}")
print(f"Network output: {output}")
```

### 2. Loss Functions
Loss functions measure how far our predictions are from the actual values.

```python
def mean_squared_error(y_true, y_pred):
    """Mean Squared Error loss function"""
    return np.mean((y_true - y_pred)**2)

def mean_absolute_error(y_true, y_pred):
    """Mean Absolute Error loss function"""
    return np.mean(np.abs(y_true - y_pred))

def binary_crossentropy(y_true, y_pred):
    """Binary Cross-Entropy loss function"""
    epsilon = 1e-15  # Small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example usage
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])

print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"BCE: {binary_crossentropy(y_true, y_pred):.4f}")
```

## Activation Functions

Activation functions introduce non-linearity into neural networks.

### 1. Sigmoid Function
f(x) = 1 / (1 + e^(-x))

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Plot sigmoid function
x = np.linspace(-6, 6, 100)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Sigmoid Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, sigmoid_derivative(x), label='Sigmoid Derivative', linewidth=2)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Sigmoid Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2. ReLU (Rectified Linear Unit)
f(x) = max(0, x)

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Plot ReLU function
x = np.linspace(-3, 3, 100)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, relu(x), label='ReLU', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('ReLU Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, relu_derivative(x), label='ReLU Derivative', linewidth=2)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('ReLU Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3. Tanh Function
f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Plot Tanh function
x = np.linspace(-3, 3, 100)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, tanh(x), label='Tanh', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Tanh Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, tanh_derivative(x), label='Tanh Derivative', linewidth=2)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Tanh Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Loss Functions

### 1. Mean Squared Error (MSE)
Used for regression problems.

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Example: Predicting house prices
actual_prices = np.array([200000, 300000, 150000, 400000])
predicted_prices = np.array([195000, 310000, 145000, 390000])

mse = mse_loss(actual_prices, predicted_prices)
print(f"MSE Loss: ${mse:,.0f}")
```

### 2. Cross-Entropy Loss
Used for classification problems.

```python
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Example: Multi-class classification
# True labels (one-hot encoded)
y_true = np.array([[1, 0, 0],  # Class 0
                   [0, 1, 0],  # Class 1
                   [0, 0, 1]]) # Class 2

# Predicted probabilities
y_pred = np.array([[0.7, 0.2, 0.1],  # Predicted as class 0
                   [0.1, 0.8, 0.1],  # Predicted as class 1
                   [0.2, 0.3, 0.5]]) # Predicted as class 2

ce_loss = cross_entropy_loss(y_true, y_pred)
print(f"Cross-Entropy Loss: {ce_loss:.4f}")
```

## Practical Examples

### Example 1: Simple Linear Regression
```python
class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute loss
            loss = np.mean((y - y_pred)**2)
            
            # Compute gradients
            dw = -(2/n_samples) * np.dot(X.T, (y - y_pred))
            db = -(2/n_samples) * np.sum(y - y_pred)
            
            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1) * 2
y = 3 * X.flatten() + 2 + np.random.randn(100) * 0.5

# Train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_test = np.array([[0], [1], [2]])
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")
```

### Example 2: Simple Neural Network for XOR
```python
class XORNetwork:
    def __init__(self):
        # Initialize weights and biases
        self.w1 = np.array([[1, -1], [-1, 1]])  # Hidden layer weights
        self.b1 = np.array([0, 0])              # Hidden layer bias
        self.w2 = np.array([1, 1])               # Output layer weights
        self.b2 = 0                              # Output layer bias
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        # Hidden layer
        z1 = np.dot(self.w1, x) + self.b1
        a1 = self.sigmoid(z1)
        
        # Output layer
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self.sigmoid(z2)
        
        return a2
    
    def predict(self, X):
        predictions = []
        for x in X:
            pred = self.forward(x)
            predictions.append(1 if pred > 0.5 else 0)
        return np.array(predictions)

# Test XOR network
xor_net = XORNetwork()

# XOR truth table
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

predictions = xor_net.predict(X_xor)
print("XOR Predictions:")
for i, (x, true_y, pred_y) in enumerate(zip(X_xor, y_xor, predictions)):
    print(f"Input: {x}, True: {true_y}, Predicted: {pred_y}")
```

## Key Takeaways

1. **Functions are fundamental** to understanding neural networks
2. **Function composition** is how neural networks process information
3. **Activation functions** introduce non-linearity
4. **Loss functions** measure prediction accuracy
5. **Mathematical functions** translate directly to code

## Next Steps

Now that you understand functions, you're ready to learn about:
- **Derivatives**: How functions change
- **Gradients**: Multi-dimensional derivatives
- **Backpropagation**: Using gradients to train neural networks

Functions are the building blocks of everything in neural networks. Every operation, from simple addition to complex transformations, can be understood as a function. Master functions, and you'll have a solid foundation for understanding how neural networks work!
