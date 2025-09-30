# Gradients: Multi-Dimensional Derivatives and Backpropagation

## Table of Contents
1. [What are Gradients?](#what-are-gradients)
2. [Gradient Vector](#gradient-vector)
3. [Gradient Descent](#gradient-descent)
4. [Backpropagation](#backpropagation)
5. [Computational Graphs](#computational-graphs)
6. [Gradient Flow](#gradient-flow)
7. [Advanced Gradient Techniques](#advanced-gradient-techniques)
8. [Practical Examples](#practical-examples)

## What are Gradients?

A **gradient** is a vector of partial derivatives. It tells us the direction of steepest ascent for a function. In neural networks, gradients tell us how to adjust each parameter to minimize the loss function.

### Mathematical Definition
For a function f(x₁, x₂, ..., xₙ), the gradient is:
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

### Intuitive Understanding
Think of standing on a mountain:
- The gradient points in the direction of steepest ascent
- The negative gradient points in the direction of steepest descent
- The magnitude tells us how steep the slope is

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: f(x, y) = x² + y²
def f(x, y):
    return x**2 + y**2

def gradient_f(x, y):
    """Gradient of f(x, y) = x² + y²"""
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

# Visualize gradient field
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Calculate gradients at each point
U, V = np.zeros_like(X), np.zeros_like(Y)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad = gradient_f(X[i, j], Y[i, j])
        U[i, j] = grad[0]
        V[i, j] = grad[1]

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=20, colors='black', alpha=0.3)
plt.quiver(X, Y, U, V, alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Field of f(x, y) = x² + y²')
plt.colorbar()
plt.grid(True, alpha=0.3)
plt.show()
```

## Gradient Vector

### Properties of Gradients
1. **Direction**: Points toward steepest ascent
2. **Magnitude**: Indicates rate of change
3. **Orthogonality**: Gradient is perpendicular to level curves

```python
def visualize_gradient_properties():
    """Demonstrate gradient properties"""
    
    # Function: f(x, y) = x² + 2y²
    def f(x, y):
        return x**2 + 2*y**2
    
    def gradient_f(x, y):
        return np.array([2*x, 4*y])
    
    # Create grid
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    # Plot level curves and gradient
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contour(X, Y, Z, levels=15, colors='black', alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Level Curves')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.contour(X, Y, Z, levels=15, colors='black', alpha=0.3)
    
    # Plot gradient vectors at selected points
    points = np.array([[1, 0.5], [0, 1], [-1, -0.5], [0.5, -1]])
    for point in points:
        x_p, y_p = point
        grad = gradient_f(x_p, y_p)
        plt.quiver(x_p, y_p, grad[0], grad[1], scale=20, color='red', alpha=0.8)
        plt.plot(x_p, y_p, 'ro', markersize=5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Vectors')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

visualize_gradient_properties()
```

## Gradient Descent

Gradient descent is an optimization algorithm that uses gradients to find the minimum of a function.

### Algorithm
1. Start at some point x₀
2. Compute gradient ∇f(x₀)
3. Move in the opposite direction: x₁ = x₀ - α∇f(x₀)
4. Repeat until convergence

```python
def gradient_descent_example():
    """Demonstrate gradient descent on a simple function"""
    
    # Function: f(x, y) = (x-1)² + (y-2)²
    def f(x, y):
        return (x-1)**2 + (y-2)**2
    
    def gradient_f(x, y):
        return np.array([2*(x-1), 2*(y-2)])
    
    # Gradient descent parameters
    learning_rate = 0.1
    max_iterations = 50
    tolerance = 1e-6
    
    # Starting point
    x = np.array([-2.0, -1.0])
    path = [x.copy()]
    
    print(f"Starting point: ({x[0]:.3f}, {x[1]:.3f})")
    print(f"Initial function value: {f(x[0], x[1]):.6f}")
    
    for i in range(max_iterations):
        # Compute gradient
        grad = gradient_f(x[0], x[1])
        
        # Update position
        x_new = x - learning_rate * grad
        path.append(x_new.copy())
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tolerance:
            print(f"Converged after {i+1} iterations")
            break
        
        x = x_new
        
        if i % 10 == 0:
            print(f"Iteration {i}: x=({x[0]:.3f}, {x[1]:.3f}), f(x)={f(x[0], x[1]):.6f}")
    
    print(f"Final point: ({x[0]:.3f}, {x[1]:.3f})")
    print(f"Final function value: {f(x[0], x[1]):.6f}")
    
    # Visualize the path
    path = np.array(path)
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-2, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=20, colors='black', alpha=0.3)
    plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4)
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start')
    plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8, label='End')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

gradient_descent_example()
```

## Backpropagation

Backpropagation is the algorithm used to compute gradients in neural networks. It uses the chain rule to propagate gradients backward through the network.

### Forward Pass
Compute activations layer by layer from input to output.

### Backward Pass
Compute gradients layer by layer from output to input.

```python
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward pass through the network"""
        self.activations = [X.T]  # Store activations for backprop
        self.z_values = []        # Store pre-activation values
        
        current_input = X.T
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], current_input) + self.biases[i]
            self.z_values.append(z)
            
            a = self.sigmoid(z)
            self.activations.append(a)
            current_input = a
        
        return self.activations[-1].T
    
    def backward(self, X, y, output):
        """Backward pass to compute gradients"""
        m = X.shape[0]
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        dz = output.T - y.reshape(-1, 1)
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Compute gradients for current layer
            dW[i] = (1/m) * np.dot(dz, self.activations[i].T)
            db[i] = (1/m) * np.sum(dz, axis=1, keepdims=True)
            
            # Propagate gradient to previous layer
            if i > 0:
                dz = np.dot(self.weights[i].T, dz) * self.sigmoid_derivative(self.z_values[i-1])
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs=1000):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = np.mean((y.reshape(-1, 1) - output)**2)
            
            # Backward pass
            dW, db = self.backward(X, y, output)
            
            # Update parameters
            self.update_parameters(dW, db)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Test the neural network
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR problem

nn = NeuralNetwork([2, 4, 1], learning_rate=0.5)
nn.train(X, y, epochs=1000)

# Test predictions
predictions = nn.forward(X)
print("\nXOR Predictions:")
for i, (x, true_y, pred_y) in enumerate(zip(X, y, predictions)):
    print(f"Input: {x}, True: {true_y}, Predicted: {pred_y[0]:.4f}")
```

## Computational Graphs

Computational graphs visualize how operations flow through a neural network, making it easier to understand gradient computation.

```python
class ComputationalGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
    
    def add_node(self, name, operation=None):
        self.nodes.append({'name': name, 'operation': operation})
    
    def add_edge(self, from_node, to_node):
        self.edges.append((from_node, to_node))
    
    def visualize(self):
        """Simple visualization of computational graph"""
        print("Computational Graph:")
        for node in self.nodes:
            print(f"Node: {node['name']}")
            if node['operation']:
                print(f"  Operation: {node['operation']}")
        
        print("\nEdges:")
        for edge in self.edges:
            print(f"  {edge[0]} -> {edge[1]}")

# Example: Simple computational graph for f(x) = (x + 1) * (x - 1)
def create_simple_graph():
    graph = ComputationalGraph()
    
    # Add nodes
    graph.add_node('x', 'input')
    graph.add_node('x_plus_1', '+')
    graph.add_node('x_minus_1', '-')
    graph.add_node('multiply', '*')
    graph.add_node('output', 'output')
    
    # Add edges
    graph.add_edge('x', 'x_plus_1')
    graph.add_edge('x', 'x_minus_1')
    graph.add_edge('x_plus_1', 'multiply')
    graph.add_edge('x_minus_1', 'multiply')
    graph.add_edge('multiply', 'output')
    
    return graph

# Create and visualize graph
graph = create_simple_graph()
graph.visualize()
```

## Gradient Flow

Understanding gradient flow is crucial for training deep networks. Gradients can vanish or explode as they propagate backward.

### Gradient Vanishing Problem
In deep networks, gradients can become very small, making it hard to train early layers.

```python
def demonstrate_gradient_vanishing():
    """Demonstrate gradient vanishing problem"""
    
    # Create a deep network with sigmoid activations
    layer_sizes = [1, 10, 10, 10, 10, 1]
    weights = []
    
    # Initialize weights
    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1
        weights.append(w)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    # Forward pass
    x = np.array([[1.0]])
    activations = [x.T]
    z_values = []
    
    for i in range(len(weights)):
        z = np.dot(weights[i], activations[-1])
        z_values.append(z)
        a = sigmoid(z)
        activations.append(a)
    
    # Backward pass - compute gradient magnitudes
    gradient_magnitudes = []
    
    # Start with output gradient
    dz = np.ones_like(activations[-1])
    
    for i in reversed(range(len(weights))):
        # Compute gradient magnitude
        gradient_magnitude = np.mean(np.abs(dz))
        gradient_magnitudes.append(gradient_magnitude)
        
        # Propagate gradient
        if i > 0:
            dz = np.dot(weights[i].T, dz) * sigmoid_derivative(z_values[i-1])
    
    # Plot gradient magnitudes
    layers = list(range(len(gradient_magnitudes)))
    plt.figure(figsize=(10, 6))
    plt.plot(layers, gradient_magnitudes, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Layer (from output to input)')
    plt.ylabel('Average Gradient Magnitude')
    plt.title('Gradient Vanishing Problem')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Gradient magnitudes:")
    for i, mag in enumerate(gradient_magnitudes):
        print(f"Layer {i}: {mag:.6f}")

demonstrate_gradient_vanishing()
```

### Solutions to Gradient Problems

```python
def demonstrate_gradient_solutions():
    """Demonstrate solutions to gradient problems"""
    
    # 1. ReLU activation (helps with vanishing gradients)
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    # 2. Xavier/Glorot initialization
    def xavier_init(fan_in, fan_out):
        return np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / (fan_in + fan_out))
    
    # 3. Batch normalization (simplified)
    def batch_norm(x, gamma=1, beta=0):
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        normalized = (x - mean) / np.sqrt(var + 1e-8)
        return gamma * normalized + beta
    
    # Create network with better initialization and ReLU
    layer_sizes = [1, 10, 10, 10, 10, 1]
    weights = []
    
    for i in range(len(layer_sizes) - 1):
        w = xavier_init(layer_sizes[i], layer_sizes[i+1])
        weights.append(w)
    
    # Forward pass with ReLU
    x = np.array([[1.0]])
    activations = [x.T]
    z_values = []
    
    for i in range(len(weights)):
        z = np.dot(weights[i], activations[-1])
        z_values.append(z)
        a = relu(z)
        activations.append(a)
    
    # Backward pass
    gradient_magnitudes = []
    dz = np.ones_like(activations[-1])
    
    for i in reversed(range(len(weights))):
        gradient_magnitude = np.mean(np.abs(dz))
        gradient_magnitudes.append(gradient_magnitude)
        
        if i > 0:
            dz = np.dot(weights[i].T, dz) * relu_derivative(z_values[i-1])
    
    # Plot gradient magnitudes
    layers = list(range(len(gradient_magnitudes)))
    plt.figure(figsize=(10, 6))
    plt.plot(layers, gradient_magnitudes, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Layer (from output to input)')
    plt.ylabel('Average Gradient Magnitude')
    plt.title('Gradient Flow with ReLU and Xavier Initialization')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()

demonstrate_gradient_solutions()
```

## Advanced Gradient Techniques

### 1. Momentum
Momentum helps accelerate convergence by accumulating gradients over time.

```python
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, gradients, parameters):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in parameters]
        
        updated_params = []
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # Update velocity
            self.velocity[i] = self.momentum * self.velocity[i] + self.learning_rate * grad
            # Update parameters
            updated_params.append(param - self.velocity[i])
        
        return updated_params

# Example usage
def test_momentum_optimizer():
    # Simple quadratic function: f(x) = (x-1)²
    def f(x):
        return (x - 1)**2
    
    def gradient_f(x):
        return 2 * (x - 1)
    
    # Compare standard gradient descent vs momentum
    x_standard = np.array([-2.0])
    x_momentum = np.array([-2.0])
    
    optimizer = MomentumOptimizer(learning_rate=0.1, momentum=0.9)
    
    standard_path = [x_standard.copy()]
    momentum_path = [x_momentum.copy()]
    
    for i in range(20):
        # Standard gradient descent
        grad_standard = gradient_f(x_standard[0])
        x_standard = x_standard - 0.1 * grad_standard
        standard_path.append(x_standard.copy())
        
        # Momentum optimizer
        grad_momentum = gradient_f(x_momentum[0])
        x_momentum = optimizer.update([grad_momentum], [x_momentum])[0]
        momentum_path.append(x_momentum.copy())
    
    # Plot comparison
    iterations = range(len(standard_path))
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations, standard_path, 'b-o', label='Standard GD', linewidth=2)
    plt.plot(iterations, momentum_path, 'r-o', label='Momentum', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('x value')
    plt.title('Parameter Updates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    standard_loss = [f(x[0]) for x in standard_path]
    momentum_loss = [f(x[0]) for x in momentum_path]
    plt.plot(iterations, standard_loss, 'b-o', label='Standard GD', linewidth=2)
    plt.plot(iterations, momentum_loss, 'r-o', label='Momentum', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

test_momentum_optimizer()
```

### 2. Adam Optimizer
Adam combines momentum with adaptive learning rates.

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0     # Time step
    
    def update(self, gradients, parameters):
        self.t += 1
        
        if self.m is None:
            self.m = [np.zeros_like(p) for p in parameters]
            self.v = [np.zeros_like(p) for p in parameters]
        
        updated_params = []
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            updated_params.append(param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
        
        return updated_params

# Test Adam optimizer
def test_adam_optimizer():
    # Rosenbrock function: f(x, y) = (1-x)² + 100(y-x²)²
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rosenbrock_gradient(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])
    
    # Starting point
    x = np.array([-1.0, 1.0])
    optimizer = AdamOptimizer(learning_rate=0.01)
    
    path = [x.copy()]
    
    for i in range(1000):
        grad = rosenbrock_gradient(x[0], x[1])
        x = optimizer.update([grad], [x])[0]
        path.append(x.copy())
        
        if i % 100 == 0:
            loss = rosenbrock(x[0], x[1])
            print(f"Iteration {i}: x=({x[0]:.4f}, {x[1]:.4f}), Loss={loss:.6f}")
    
    # Plot optimization path
    path = np.array(path)
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = rosenbrock(X, Y)
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=20, colors='black', alpha=0.3)
    plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=3)
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start')
    plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8, label='End')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Adam Optimization on Rosenbrock Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

test_adam_optimizer()
```

## Practical Examples

### Example 1: Multi-Layer Perceptron with Backpropagation
```python
class MLP:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def forward(self, X):
        self.activations = [X.T]
        self.z_values = []
        
        current_input = X.T
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], current_input) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)
            current_input = a
        
        # Output layer with softmax
        z = np.dot(self.weights[-1], current_input) + self.biases[-1]
        self.z_values.append(z)
        a = self.softmax(z)
        self.activations.append(a)
        
        return self.activations[-1].T
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Convert y to one-hot encoding
        y_onehot = np.zeros((m, self.layer_sizes[-1]))
        y_onehot[np.arange(m), y] = 1
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient (softmax + cross-entropy)
        dz = output.T - y_onehot.T
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            dW[i] = (1/m) * np.dot(dz, self.activations[i].T)
            db[i] = (1/m) * np.sum(dz, axis=1, keepdims=True)
            
            if i > 0:
                dz = np.dot(self.weights[i].T, dz) * self.relu_derivative(self.z_values[i-1])
        
        return dW, db
    
    def update_parameters(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            y_onehot = np.zeros((X.shape[0], self.layer_sizes[-1]))
            y_onehot[np.arange(X.shape[0]), y] = 1
            loss = -np.mean(np.sum(y_onehot * np.log(output + 1e-15), axis=1))
            
            # Backward pass
            dW, db = self.backward(X, y, output)
            
            # Update parameters
            self.update_parameters(dW, db)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Test MLP on iris dataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create and train MLP
mlp = MLP([4, 8, 3], learning_rate=0.1)
mlp.train(X, y, epochs=1000)

# Test predictions
predictions = mlp.predict(X)
accuracy = np.mean(predictions == y)
print(f"Training Accuracy: {accuracy:.4f}")
```

### Example 2: Convolutional Neural Network (Simplified)
```python
class SimpleCNN:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Simple CNN: Conv -> ReLU -> MaxPool -> Dense -> Softmax
        self.conv_filter = np.random.randn(3, 3) * 0.1
        self.dense_weights = np.random.randn(10, 10) * 0.1
        self.dense_bias = np.zeros((10, 1))
    
    def conv2d(self, image, filter):
        """Simple 2D convolution"""
        h, w = image.shape
        fh, fw = filter.shape
        
        output = np.zeros((h - fh + 1, w - fw + 1))
        for i in range(h - fh + 1):
            for j in range(w - fw + 1):
                output[i, j] = np.sum(image[i:i+fh, j:j+fw] * filter)
        
        return output
    
    def maxpool2d(self, image, pool_size=2):
        """Simple 2D max pooling"""
        h, w = image.shape
        output_h = h // pool_size
        output_w = w // pool_size
        
        output = np.zeros((output_h, output_w))
        for i in range(output_h):
            for j in range(output_w):
                start_i = i * pool_size
                start_j = j * pool_size
                output[i, j] = np.max(image[start_i:start_i+pool_size, 
                                           start_j:start_j+pool_size])
        
        return output
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, image):
        # Convolution
        conv_out = self.conv2d(image, self.conv_filter)
        
        # ReLU
        relu_out = self.relu(conv_out)
        
        # Max pooling
        pool_out = self.maxpool2d(relu_out)
        
        # Flatten
        flattened = pool_out.flatten().reshape(-1, 1)
        
        # Dense layer
        dense_out = np.dot(self.dense_weights, flattened) + self.dense_bias
        
        # Softmax
        output = self.softmax(dense_out)
        
        return output, conv_out, relu_out, pool_out, flattened
    
    def backward(self, image, target, forward_outputs):
        output, conv_out, relu_out, pool_out, flattened = forward_outputs
        
        # Convert target to one-hot
        target_onehot = np.zeros((10, 1))
        target_onehot[target] = 1
        
        # Output layer gradient
        dz_output = output - target_onehot
        
        # Dense layer gradients
        dW_dense = np.dot(dz_output, flattened.T)
        db_dense = dz_output
        
        # Propagate gradient back to conv layer
        dz_dense = np.dot(self.dense_weights.T, dz_output)
        
        # Reshape gradient for pooling layer
        dz_pool = dz_dense.reshape(pool_out.shape)
        
        # Max pooling backward pass (simplified)
        dz_relu = np.zeros_like(relu_out)
        h, w = relu_out.shape
        pool_size = 2
        
        for i in range(h // pool_size):
            for j in range(w // pool_size):
                start_i = i * pool_size
                start_j = j * pool_size
                region = relu_out[start_i:start_i+pool_size, start_j:start_j+pool_size]
                max_idx = np.unravel_index(np.argmax(region), region.shape)
                dz_relu[start_i + max_idx[0], start_j + max_idx[1]] = dz_pool[i, j]
        
        # ReLU backward pass
        dz_conv = dz_relu * (conv_out > 0).astype(float)
        
        # Convolution backward pass
        dW_conv = np.zeros_like(self.conv_filter)
        for i in range(self.conv_filter.shape[0]):
            for j in range(self.conv_filter.shape[1]):
                dW_conv[i, j] = np.sum(image[i:i+dz_conv.shape[0], j:j+dz_conv.shape[1]] * dz_conv)
        
        return dW_conv, dW_dense, db_dense
    
    def update_parameters(self, dW_conv, dW_dense, db_dense):
        self.conv_filter -= self.learning_rate * dW_conv
        self.dense_weights -= self.learning_rate * dW_dense
        self.dense_bias -= self.learning_rate * db_dense
    
    def train_step(self, image, target):
        # Forward pass
        output, *forward_outputs = self.forward(image)
        
        # Backward pass
        dW_conv, dW_dense, db_dense = self.backward(image, target, forward_outputs)
        
        # Update parameters
        self.update_parameters(dW_conv, dW_dense, db_dense)
        
        # Compute loss
        loss = -np.log(output[target] + 1e-15)
        
        return loss

# Test simple CNN
def test_simple_cnn():
    # Create random image and target
    image = np.random.randn(10, 10)
    target = 5
    
    cnn = SimpleCNN(learning_rate=0.01)
    
    # Train for a few steps
    for i in range(100):
        loss = cnn.train_step(image, target)
        if i % 20 == 0:
            print(f"Step {i}, Loss: {loss:.6f}")
    
    # Test prediction
    output, _, _, _, _ = cnn.forward(image)
    prediction = np.argmax(output)
    print(f"Target: {target}, Prediction: {prediction}")

test_simple_cnn()
```

## Key Takeaways

1. **Gradients are vectors** of partial derivatives that point toward steepest ascent
2. **Gradient descent** uses negative gradients to find function minima
3. **Backpropagation** computes gradients efficiently using the chain rule
4. **Computational graphs** help visualize gradient flow
5. **Gradient problems** (vanishing/exploding) can be solved with proper techniques
6. **Advanced optimizers** (Adam, Momentum) improve training efficiency

## Why Gradients Matter in Neural Networks

1. **Parameter Updates**: Gradients tell us how to adjust weights and biases
2. **Learning**: Gradients enable the network to learn from data
3. **Optimization**: Gradients guide the search for optimal parameters
4. **Efficiency**: Backpropagation makes gradient computation feasible

## Next Steps

Now that you understand gradients, you're ready to:
- **Implement advanced optimizers** (Adam, RMSprop, etc.)
- **Handle gradient problems** (vanishing/exploding gradients)
- **Build deeper networks** with proper gradient flow
- **Understand modern architectures** (ResNet, Transformer, etc.)

Gradients are the mathematical engine that powers neural network learning. They transform static functions into adaptive systems that can learn from data!
