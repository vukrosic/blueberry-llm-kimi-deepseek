# Derivatives: Understanding How Functions Change

## Table of Contents
1. [What are Derivatives?](#what-are-derivatives)
2. [The Concept of Rate of Change](#the-concept-of-rate-of-change)
3. [Calculating Derivatives](#calculating-derivatives)
4. [Common Derivative Rules](#common-derivative-rules)
5. [Derivatives of Neural Network Functions](#derivatives-of-neural-network-functions)
6. [Chain Rule](#chain-rule)
7. [Partial Derivatives](#partial-derivatives)
8. [Practical Examples](#practical-examples)

## What are Derivatives?

A **derivative** measures how a function changes as its input changes. It tells us the **instantaneous rate of change** of a function at any point.

### Intuitive Understanding
Think of driving a car:
- Your position is a function of time: position(t)
- Your speed is the derivative of position: speed = d(position)/dt
- Speed tells you how fast your position is changing

### Mathematical Definition
The derivative of f(x) at point x is:
```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

### Visual Representation
```python
import numpy as np
import matplotlib.pyplot as plt

# Define a function
def f(x):
    return x**2

# Define its derivative
def f_prime(x):
    return 2*x

# Plot function and its derivative
x = np.linspace(-3, 3, 100)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, f(x), label='f(x) = x²', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, f_prime(x), label="f'(x) = 2x", linewidth=2)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Derivative')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## The Concept of Rate of Change

### Average Rate of Change
The average rate of change between two points:
```
Average rate = (f(b) - f(a)) / (b - a)
```

### Instantaneous Rate of Change
The derivative gives us the instantaneous rate of change at a specific point.

```python
def average_rate_of_change(f, a, b):
    """Calculate average rate of change between a and b"""
    return (f(b) - f(a)) / (b - a)

def numerical_derivative(f, x, h=1e-5):
    """Calculate derivative numerically"""
    return (f(x + h) - f(x)) / h

# Example: f(x) = x³
def cubic_function(x):
    return x**3

# Calculate average rate of change
a, b = 1, 2
avg_rate = average_rate_of_change(cubic_function, a, b)
print(f"Average rate of change from {a} to {b}: {avg_rate}")

# Calculate instantaneous rate of change (derivative)
x_point = 1.5
instant_rate = numerical_derivative(cubic_function, x_point)
print(f"Instantaneous rate of change at x={x_point}: {instant_rate}")

# Compare with analytical derivative: f'(x) = 3x²
analytical_derivative = 3 * x_point**2
print(f"Analytical derivative at x={x_point}: {analytical_derivative}")
```

## Calculating Derivatives

### Method 1: Numerical Differentiation
Using the limit definition with a small h value.

```python
def numerical_derivative(f, x, h=1e-5):
    """Calculate derivative using numerical method"""
    return (f(x + h) - f(x)) / h

# Test with different functions
def test_function(x):
    return 2*x**2 + 3*x + 1

def analytical_derivative(x):
    return 4*x + 3

# Compare numerical vs analytical
x_values = [0, 1, 2, 3]
for x in x_values:
    numerical = numerical_derivative(test_function, x)
    analytical = analytical_derivative(x)
    print(f"x={x}: Numerical={numerical:.6f}, Analytical={analytical:.6f}")
```

### Method 2: Symbolic Differentiation
Using mathematical rules to find exact derivatives.

```python
# Example: Find derivative of f(x) = x³ + 2x² + 5x + 1
# f'(x) = 3x² + 4x + 5

def polynomial_function(x):
    return x**3 + 2*x**2 + 5*x + 1

def polynomial_derivative(x):
    return 3*x**2 + 4*x + 5

# Verify with numerical method
x_test = 2
numerical = numerical_derivative(polynomial_function, x_test)
analytical = polynomial_derivative(x_test)
print(f"At x={x_test}:")
print(f"Numerical derivative: {numerical:.6f}")
print(f"Analytical derivative: {analytical:.6f}")
print(f"Difference: {abs(numerical - analytical):.8f}")
```

## Table of Common Derivatives

Here's a comprehensive table of derivatives you'll encounter in neural networks:

| Function | Derivative | Notes |
|----------|------------|-------|
| **Power Functions** |
| f(x) = c (constant) | f'(x) = 0 | Constant rule |
| f(x) = x | f'(x) = 1 | Identity function |
| f(x) = x² | f'(x) = 2x | Power rule |
| f(x) = x³ | f'(x) = 3x² | Power rule |
| f(x) = xⁿ | f'(x) = nxⁿ⁻¹ | General power rule |
| f(x) = √x = x^(1/2) | f'(x) = 1/(2√x) | Square root |
| f(x) = 1/x = x^(-1) | f'(x) = -1/x² | Reciprocal |
| **Exponential & Logarithmic** |
| f(x) = eˣ | f'(x) = eˣ | Natural exponential |
| f(x) = aˣ | f'(x) = aˣ ln(a) | General exponential |
| f(x) = ln(x) | f'(x) = 1/x | Natural logarithm |
| f(x) = log_a(x) | f'(x) = 1/(x ln(a)) | General logarithm |
| **Trigonometric Functions** |
| f(x) = sin(x) | f'(x) = cos(x) | Sine |
| f(x) = cos(x) | f'(x) = -sin(x) | Cosine |
| f(x) = tan(x) | f'(x) = sec²(x) = 1/cos²(x) | Tangent |
| f(x) = cot(x) | f'(x) = -csc²(x) = -1/sin²(x) | Cotangent |
| f(x) = sec(x) | f'(x) = sec(x)tan(x) | Secant |
| f(x) = csc(x) | f'(x) = -csc(x)cot(x) | Cosecant |
| **Neural Network Functions** |
| f(x) = sigmoid(x) = 1/(1+e^(-x)) | f'(x) = f(x)(1-f(x)) | Sigmoid |
| f(x) = tanh(x) | f'(x) = 1 - tanh²(x) | Hyperbolic tangent |
| f(x) = ReLU(x) = max(0,x) | f'(x) = {1 if x>0, 0 if x≤0} | Rectified Linear Unit |
| f(x) = Leaky ReLU(x) | f'(x) = {1 if x>0, α if x≤0} | Leaky ReLU (α≈0.01) |
| f(x) = softmax(x) | Complex (see softmax section) | Softmax |

## Common Derivative Rules

### 1. Power Rule
If f(x) = xⁿ, then f'(x) = nxⁿ⁻¹

#### Step-by-Step Examples

**Example 1: f(x) = x²**
- Using power rule: f'(x) = 2x^(2-1) = 2x¹ = 2x
- Verification: f'(x) = 2x

**Example 2: f(x) = x³**
- Using power rule: f'(x) = 3x^(3-1) = 3x²
- Verification: f'(x) = 3x²

**Example 3: f(x) = x⁴**
- Using power rule: f'(x) = 4x^(4-1) = 4x³
- Verification: f'(x) = 4x³

**Example 4: f(x) = √x = x^(1/2)**
- Using power rule: f'(x) = (1/2)x^((1/2)-1) = (1/2)x^(-1/2) = 1/(2√x)
- Verification: f'(x) = 1/(2√x)

**Example 5: f(x) = 1/x = x^(-1)**
- Using power rule: f'(x) = (-1)x^(-1-1) = (-1)x^(-2) = -1/x²
- Verification: f'(x) = -1/x²

```python
def power_rule_examples():
    """Demonstrate power rule"""
    examples = [
        ("f(x) = x²", lambda x: x**2, lambda x: 2*x),
        ("f(x) = x³", lambda x: x**3, lambda x: 3*x**2),
        ("f(x) = x⁴", lambda x: x**4, lambda x: 4*x**3),
        ("f(x) = √x = x^(1/2)", lambda x: x**0.5, lambda x: 0.5*x**(-0.5)),
        ("f(x) = 1/x = x^(-1)", lambda x: 1/x, lambda x: -1*x**(-2))
    ]
    
    x_test = 2
    for name, func, deriv in examples:
        if x_test > 0 or "√" not in name:  # Avoid sqrt of negative
            numerical = numerical_derivative(func, x_test)
            analytical = deriv(x_test)
            print(f"{name}:")
            print(f"  Numerical: {numerical:.6f}")
            print(f"  Analytical: {analytical:.6f}")
            print()

power_rule_examples()
```

### 2. Constant Multiple Rule
If f(x) = c·g(x), then f'(x) = c·g'(x)

#### Step-by-Step Examples

**Example: f(x) = 5x²**

Step 1: Identify the constant and the function
- Constant: c = 5
- Function: g(x) = x²

Step 2: Find g'(x)
- g'(x) = 2x (using power rule)

Step 3: Apply constant multiple rule
- f'(x) = c·g'(x) = 5·(2x) = 10x

**Verification:**
- f(x) = 5x²
- f'(x) = 10x ✓

**Example: f(x) = -3x³**

Step 1: Identify the constant and the function
- Constant: c = -3
- Function: g(x) = x³

Step 2: Find g'(x)
- g'(x) = 3x² (using power rule)

Step 3: Apply constant multiple rule
- f'(x) = c·g'(x) = (-3)·(3x²) = -9x²

**Verification:**
- f(x) = -3x³
- f'(x) = -9x² ✓

```python
def constant_multiple_examples():
    """Demonstrate constant multiple rule"""
    def g(x):
        return x**2
    
    def g_prime(x):
        return 2*x
    
    # f(x) = 5x², f'(x) = 5·2x = 10x
    def f(x):
        return 5 * g(x)
    
    def f_prime(x):
        return 5 * g_prime(x)
    
    x_test = 3
    numerical = numerical_derivative(f, x_test)
    analytical = f_prime(x_test)
    
    print(f"f(x) = 5x²:")
    print(f"Numerical derivative: {numerical:.6f}")
    print(f"Analytical derivative: {analytical:.6f}")
    print(f"Expected: 10x = {10*x_test}")

constant_multiple_examples()
```

### 3. Sum Rule
If f(x) = g(x) + h(x), then f'(x) = g'(x) + h'(x)

#### Step-by-Step Examples

**Example: f(x) = x² + 3x**

Step 1: Identify the functions
- g(x) = x²
- h(x) = 3x

Step 2: Find individual derivatives
- g'(x) = 2x (power rule)
- h'(x) = 3 (constant multiple rule: 3·1 = 3)

Step 3: Apply sum rule
- f'(x) = g'(x) + h'(x) = 2x + 3

**Verification:**
- f(x) = x² + 3x
- f'(x) = 2x + 3 ✓

**Example: f(x) = x³ + 2x² + 5x + 1**

Step 1: Identify the functions
- g(x) = x³
- h(x) = 2x²
- i(x) = 5x
- j(x) = 1

Step 2: Find individual derivatives
- g'(x) = 3x² (power rule)
- h'(x) = 4x (constant multiple rule: 2·2x = 4x)
- i'(x) = 5 (constant multiple rule: 5·1 = 5)
- j'(x) = 0 (constant rule)

Step 3: Apply sum rule
- f'(x) = g'(x) + h'(x) + i'(x) + j'(x) = 3x² + 4x + 5 + 0 = 3x² + 4x + 5

**Verification:**
- f(x) = x³ + 2x² + 5x + 1
- f'(x) = 3x² + 4x + 5 ✓

```python
def sum_rule_examples():
    """Demonstrate sum rule"""
    def g(x):
        return x**2
    
    def h(x):
        return 3*x
    
    def f(x):
        return g(x) + h(x)  # f(x) = x² + 3x
    
    def f_prime(x):
        return 2*x + 3  # f'(x) = 2x + 3
    
    x_test = 2
    numerical = numerical_derivative(f, x_test)
    analytical = f_prime(x_test)
    
    print(f"f(x) = x² + 3x:")
    print(f"Numerical derivative: {numerical:.6f}")
    print(f"Analytical derivative: {analytical:.6f}")

sum_rule_examples()
```

### 4. Product Rule
If f(x) = g(x)·h(x), then f'(x) = g'(x)·h(x) + g(x)·h'(x)

#### Step-by-Step Examples

**Example: f(x) = x²(x + 1)**

Step 1: Identify the functions
- g(x) = x²
- h(x) = x + 1

Step 2: Find individual derivatives
- g'(x) = 2x (power rule)
- h'(x) = 1 (sum rule: derivative of x is 1, derivative of 1 is 0)

Step 3: Apply product rule
- f'(x) = g'(x)·h(x) + g(x)·h'(x)
- f'(x) = (2x)·(x + 1) + (x²)·(1)
- f'(x) = 2x(x + 1) + x²
- f'(x) = 2x² + 2x + x²
- f'(x) = 3x² + 2x

**Verification by expanding first:**
- f(x) = x²(x + 1) = x³ + x²
- f'(x) = 3x² + 2x ✓

**Example: f(x) = (2x + 3)(x² - 1)**

Step 1: Identify the functions
- g(x) = 2x + 3
- h(x) = x² - 1

Step 2: Find individual derivatives
- g'(x) = 2 (sum rule: derivative of 2x is 2, derivative of 3 is 0)
- h'(x) = 2x (sum rule: derivative of x² is 2x, derivative of -1 is 0)

Step 3: Apply product rule
- f'(x) = g'(x)·h(x) + g(x)·h'(x)
- f'(x) = (2)·(x² - 1) + (2x + 3)·(2x)
- f'(x) = 2(x² - 1) + (2x + 3)(2x)
- f'(x) = 2x² - 2 + 4x² + 6x
- f'(x) = 6x² + 6x - 2

```python
def product_rule_examples():
    """Demonstrate product rule"""
    def g(x):
        return x**2
    
    def h(x):
        return x + 1
    
    def f(x):
        return g(x) * h(x)  # f(x) = x²(x + 1) = x³ + x²
    
    def f_prime(x):
        return 3*x**2 + 2*x  # f'(x) = 3x² + 2x
    
    x_test = 2
    numerical = numerical_derivative(f, x_test)
    analytical = f_prime(x_test)
    
    print(f"f(x) = x²(x + 1):")
    print(f"Numerical derivative: {numerical:.6f}")
    print(f"Analytical derivative: {analytical:.6f}")

product_rule_examples()
```

### 5. Chain Rule
If f(x) = g(h(x)), then f'(x) = g'(h(x))·h'(x)

#### Step-by-Step Examples

**Example: f(x) = (x² + 1)³**

Step 1: Identify the inner and outer functions
- Inner function: h(x) = x² + 1
- Outer function: g(u) = u³ (where u = h(x))

Step 2: Find individual derivatives
- h'(x) = 2x (sum rule: derivative of x² is 2x, derivative of 1 is 0)
- g'(u) = 3u² (power rule)

Step 3: Apply chain rule
- f'(x) = g'(h(x))·h'(x)
- f'(x) = 3(h(x))²·(2x)
- f'(x) = 3(x² + 1)²·(2x)
- f'(x) = 6x(x² + 1)²

**Verification by expanding first:**
- f(x) = (x² + 1)³ = (x² + 1)(x² + 1)(x² + 1)
- Expanding: f(x) = x⁶ + 3x⁴ + 3x² + 1
- f'(x) = 6x⁵ + 12x³ + 6x = 6x(x⁴ + 2x² + 1) = 6x(x² + 1)² ✓

**Example: f(x) = √(x² + 4)**

Step 1: Identify the inner and outer functions
- Inner function: h(x) = x² + 4
- Outer function: g(u) = √u = u^(1/2) (where u = h(x))

Step 2: Find individual derivatives
- h'(x) = 2x (sum rule: derivative of x² is 2x, derivative of 4 is 0)
- g'(u) = (1/2)u^(-1/2) = 1/(2√u) (power rule)

Step 3: Apply chain rule
- f'(x) = g'(h(x))·h'(x)
- f'(x) = (1/(2√(x² + 4)))·(2x)
- f'(x) = 2x/(2√(x² + 4))
- f'(x) = x/√(x² + 4)

```python
def chain_rule_examples():
    """Demonstrate chain rule"""
    def h(x):
        return x**2 + 1
    
    def g(u):
        return u**3
    
    def f(x):
        return g(h(x))  # f(x) = (x² + 1)³
    
    def f_prime(x):
        return 3 * (x**2 + 1)**2 * 2*x  # f'(x) = 3(x²+1)²·2x
    
    x_test = 1
    numerical = numerical_derivative(f, x_test)
    analytical = f_prime(x_test)
    
    print(f"f(x) = (x² + 1)³:")
    print(f"Numerical derivative: {numerical:.6f}")
    print(f"Analytical derivative: {analytical:.6f}")

chain_rule_examples()
```

## Derivatives of Neural Network Functions

### 1. Sigmoid Function
f(x) = 1 / (1 + e^(-x))

#### Step-by-Step Derivative Calculation

To find the derivative of sigmoid, we'll use the quotient rule and chain rule.

**Step 1: Rewrite the function**
- f(x) = 1 / (1 + e^(-x))
- Let u = 1 + e^(-x), so f(x) = 1/u

**Step 2: Apply quotient rule**
- f'(x) = (0·u - 1·u') / u² = -u' / u²

**Step 3: Find u' using chain rule**
- u = 1 + e^(-x)
- u' = 0 + e^(-x) · (-1) = -e^(-x)

**Step 4: Substitute back**
- f'(x) = -(-e^(-x)) / (1 + e^(-x))²
- f'(x) = e^(-x) / (1 + e^(-x))²

**Step 5: Simplify**
- f'(x) = e^(-x) / (1 + e^(-x))²
- f'(x) = [e^(-x) / (1 + e^(-x))] · [1 / (1 + e^(-x))]
- f'(x) = [1 / (1 + e^(-x))] · [e^(-x) / (1 + e^(-x))]
- f'(x) = f(x) · [e^(-x) / (1 + e^(-x))]

**Step 6: Further simplification**
- Notice that e^(-x) / (1 + e^(-x)) = 1 - 1/(1 + e^(-x)) = 1 - f(x)
- Therefore: f'(x) = f(x) · (1 - f(x))

**Final Result: f'(x) = f(x)(1 - f(x))**

#### Hand Calculation Examples

**Example: Find sigmoid derivative at x = 0**

Step 1: Calculate f(0)
- f(0) = 1 / (1 + e^(-0)) = 1 / (1 + e^0) = 1 / (1 + 1) = 1/2 = 0.5

Step 2: Calculate f'(0)
- f'(0) = f(0) · (1 - f(0)) = 0.5 · (1 - 0.5) = 0.5 · 0.5 = 0.25

**Example: Find sigmoid derivative at x = 1**

Step 1: Calculate f(1)
- f(1) = 1 / (1 + e^(-1)) = 1 / (1 + 1/e) = 1 / (1 + 0.368) = 1 / 1.368 ≈ 0.731

Step 2: Calculate f'(1)
- f'(1) = f(1) · (1 - f(1)) = 0.731 · (1 - 0.731) = 0.731 · 0.269 ≈ 0.197

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Plot sigmoid and its derivative
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

# Verify derivative calculation
x_test = 0
numerical = numerical_derivative(sigmoid, x_test)
analytical = sigmoid_derivative(x_test)
print(f"Sigmoid derivative at x={x_test}:")
print(f"Numerical: {numerical:.6f}")
print(f"Analytical: {analytical:.6f}")
```

### 2. ReLU Function
f(x) = max(0, x)

#### Step-by-Step Derivative Calculation

The ReLU function is piecewise defined:
- f(x) = x when x > 0
- f(x) = 0 when x ≤ 0

**For x > 0:**
- f(x) = x
- f'(x) = 1 (derivative of x is 1)

**For x < 0:**
- f(x) = 0
- f'(x) = 0 (derivative of constant is 0)

**For x = 0:**
- The function is not differentiable at x = 0 (there's a sharp corner)
- In practice, we often define f'(0) = 0 or f'(0) = 1

**Final Result: f'(x) = {1 if x > 0, 0 if x ≤ 0}**

#### Hand Calculation Examples

**Example: Find ReLU derivative at x = 2**
- Since x = 2 > 0, f'(2) = 1

**Example: Find ReLU derivative at x = -1**
- Since x = -1 < 0, f'(-1) = 0

**Example: Find ReLU derivative at x = 0**
- At x = 0, the derivative is undefined, but we typically use f'(0) = 0

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Plot ReLU and its derivative
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

#### Step-by-Step Derivative Calculation

**Method 1: Using the definition**

Step 1: Rewrite tanh using exponentials
- f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Step 2: Apply quotient rule
- Let u = e^x - e^(-x) and v = e^x + e^(-x)
- f'(x) = (u'v - uv') / v²

Step 3: Find derivatives
- u' = e^x - (-1)e^(-x) = e^x + e^(-x) = v
- v' = e^x + (-1)e^(-x) = e^x - e^(-x) = u

Step 4: Substitute
- f'(x) = (v·v - u·u) / v² = (v² - u²) / v²

Step 5: Simplify
- f'(x) = (v² - u²) / v² = 1 - (u² / v²)
- f'(x) = 1 - tanh²(x)

**Method 2: Using identity**

We can also use the identity: tanh'(x) = 1 - tanh²(x)

**Final Result: f'(x) = 1 - tanh²(x)**

#### Hand Calculation Examples

**Example: Find tanh derivative at x = 0**

Step 1: Calculate tanh(0)
- tanh(0) = (e^0 - e^(-0)) / (e^0 + e^(-0)) = (1 - 1) / (1 + 1) = 0/2 = 0

Step 2: Calculate tanh'(0)
- tanh'(0) = 1 - tanh²(0) = 1 - 0² = 1 - 0 = 1

**Example: Find tanh derivative at x = 1**

Step 1: Calculate tanh(1)
- tanh(1) = (e^1 - e^(-1)) / (e^1 + e^(-1)) = (e - 1/e) / (e + 1/e)
- tanh(1) ≈ (2.718 - 0.368) / (2.718 + 0.368) = 2.350 / 3.086 ≈ 0.762

Step 2: Calculate tanh'(1)
- tanh'(1) = 1 - tanh²(1) = 1 - (0.762)² = 1 - 0.581 ≈ 0.419

```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Plot tanh and its derivative
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

## Chain Rule

The chain rule is crucial for neural networks because it allows us to compute derivatives of composite functions.

### Mathematical Statement
If y = f(g(x)), then dy/dx = (dy/dg) × (dg/dx)

### Neural Network Application
In neural networks, we often have functions like:
f(x) = activation(linear_transformation(x))

#### Step-by-Step Chain Rule Example

**Example: Neural Network Layer with Sigmoid Activation**

Given:
- Linear transformation: z = 2x + 1
- Activation function: σ(z) = 1/(1 + e^(-z))
- Composite function: f(x) = σ(2x + 1)

**Step 1: Identify inner and outer functions**
- Inner function: h(x) = 2x + 1
- Outer function: g(z) = σ(z) = 1/(1 + e^(-z))

**Step 2: Find individual derivatives**
- h'(x) = 2 (derivative of 2x + 1)
- g'(z) = σ(z)(1 - σ(z)) (sigmoid derivative)

**Step 3: Apply chain rule**
- f'(x) = g'(h(x)) · h'(x)
- f'(x) = σ(2x + 1)(1 - σ(2x + 1)) · 2
- f'(x) = 2σ(2x + 1)(1 - σ(2x + 1))

**Step 4: Calculate at specific point (x = 1)**

Step 4a: Calculate h(1)
- h(1) = 2(1) + 1 = 3

Step 4b: Calculate σ(3)
- σ(3) = 1/(1 + e^(-3)) = 1/(1 + 0.050) = 1/1.050 ≈ 0.953

Step 4c: Calculate σ'(3)
- σ'(3) = σ(3)(1 - σ(3)) = 0.953(1 - 0.953) = 0.953(0.047) ≈ 0.045

Step 4d: Apply chain rule
- f'(1) = σ'(3) · h'(1) = 0.045 · 2 = 0.090

**Final Answer: f'(1) ≈ 0.090**

```python
def neural_network_layer_example():
    """Demonstrate chain rule in neural network layer"""
    
    # Define functions
    def linear_transform(x, w, b):
        return w * x + b
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(z):
        s = sigmoid(z)
        return s * (1 - s)
    
    # Parameters
    w = 2.0
    b = 1.0
    x = 1.5
    
    # Forward pass
    z = linear_transform(x, w, b)
    y = sigmoid(z)
    
    # Backward pass using chain rule
    # dy/dx = (dy/dz) × (dz/dx)
    dz_dx = w  # derivative of linear transformation
    dy_dz = sigmoid_derivative(z)  # derivative of sigmoid
    dy_dx = dy_dz * dz_dx  # chain rule
    
    print(f"Input x: {x}")
    print(f"Linear transformation z = {w}x + {b}: {z}")
    print(f"Sigmoid output y: {y:.6f}")
    print(f"Derivative dy/dx: {dy_dx:.6f}")
    
    # Verify with numerical derivative
    def composite_function(x):
        z = linear_transform(x, w, b)
        return sigmoid(z)
    
    numerical_deriv = numerical_derivative(composite_function, x)
    print(f"Numerical derivative: {numerical_deriv:.6f}")

neural_network_layer_example()
```

## Partial Derivatives

When we have functions of multiple variables, we use partial derivatives.

### Definition
For f(x, y), the partial derivative with respect to x is:
∂f/∂x = lim[h→0] (f(x+h, y) - f(x, y)) / h

### Example: Linear Function
f(x, y) = 2x + 3y + 1

#### Step-by-Step Partial Derivative Calculation

**Finding ∂f/∂x (partial derivative with respect to x):**

Step 1: Treat y as a constant
- f(x, y) = 2x + 3y + 1
- When taking ∂f/∂x, we treat y as constant, so 3y + 1 is constant

Step 2: Differentiate with respect to x
- ∂f/∂x = ∂/∂x(2x) + ∂/∂x(3y) + ∂/∂x(1)
- ∂f/∂x = 2 + 0 + 0 = 2

**Finding ∂f/∂y (partial derivative with respect to y):**

Step 1: Treat x as a constant
- f(x, y) = 2x + 3y + 1
- When taking ∂f/∂y, we treat x as constant, so 2x + 1 is constant

Step 2: Differentiate with respect to y
- ∂f/∂y = ∂/∂y(2x) + ∂/∂y(3y) + ∂/∂y(1)
- ∂f/∂y = 0 + 3 + 0 = 3

**Final Results:**
- ∂f/∂x = 2
- ∂f/∂y = 3

#### Hand Calculation Examples

**Example: Find partial derivatives at (x, y) = (1, 2)**

- ∂f/∂x = 2 (constant, doesn't depend on x or y)
- ∂f/∂y = 3 (constant, doesn't depend on x or y)

**Example: Find partial derivatives at (x, y) = (5, -1)**

- ∂f/∂x = 2 (still constant)
- ∂f/∂y = 3 (still constant)

```python
def linear_function_2d(x, y):
    return 2*x + 3*y + 1

def partial_x(x, y):
    return 2

def partial_y(x, y):
    return 3

# Test partial derivatives
x_test, y_test = 1, 2
print(f"f({x_test}, {y_test}) = {linear_function_2d(x_test, y_test)}")
print(f"∂f/∂x = {partial_x(x_test, y_test)}")
print(f"∂f/∂y = {partial_y(x_test, y_test)}")
```

### Example: Quadratic Function
f(x, y) = x² + 2xy + y²

#### Step-by-Step Partial Derivative Calculation

**Finding ∂f/∂x (partial derivative with respect to x):**

Step 1: Treat y as a constant
- f(x, y) = x² + 2xy + y²
- When taking ∂f/∂x, we treat y as constant

Step 2: Differentiate with respect to x
- ∂f/∂x = ∂/∂x(x²) + ∂/∂x(2xy) + ∂/∂x(y²)
- ∂f/∂x = 2x + 2y + 0 = 2x + 2y

**Finding ∂f/∂y (partial derivative with respect to y):**

Step 1: Treat x as a constant
- f(x, y) = x² + 2xy + y²
- When taking ∂f/∂y, we treat x as constant

Step 2: Differentiate with respect to y
- ∂f/∂y = ∂/∂y(x²) + ∂/∂y(2xy) + ∂/∂y(y²)
- ∂f/∂y = 0 + 2x + 2y = 2x + 2y

**Final Results:**
- ∂f/∂x = 2x + 2y
- ∂f/∂y = 2x + 2y

#### Hand Calculation Examples

**Example: Find partial derivatives at (x, y) = (1, 2)**

Step 1: Calculate ∂f/∂x
- ∂f/∂x = 2(1) + 2(2) = 2 + 4 = 6

Step 2: Calculate ∂f/∂y
- ∂f/∂y = 2(1) + 2(2) = 2 + 4 = 6

**Example: Find partial derivatives at (x, y) = (3, -1)**

Step 1: Calculate ∂f/∂x
- ∂f/∂x = 2(3) + 2(-1) = 6 - 2 = 4

Step 2: Calculate ∂f/∂y
- ∂f/∂y = 2(3) + 2(-1) = 6 - 2 = 4

```python
def quadratic_function_2d(x, y):
    return x**2 + 2*x*y + y**2

def partial_x_quad(x, y):
    return 2*x + 2*y

def partial_y_quad(x, y):
    return 2*x + 2*y

# Test partial derivatives
x_test, y_test = 1, 2
print(f"f({x_test}, {y_test}) = {quadratic_function_2d(x_test, y_test)}")
print(f"∂f/∂x = {partial_x_quad(x_test, y_test)}")
print(f"∂f/∂y = {partial_y_quad(x_test, y_test)}")
```

## Practical Examples

### Example 1: Gradient Descent for Linear Regression
```python
class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, epochs=1000):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute loss (MSE)
            loss = np.mean((y - y_pred)**2)
            
            # Compute gradients (derivatives)
            dw = -(2/n_samples) * np.dot(X.T, (y - y_pred))
            db = -(2/n_samples) * np.sum(y - y_pred)
            
            # Update parameters using gradients
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1) * 2
y = 3 * X.flatten() + 2 + np.random.randn(100) * 0.5

# Train the model
model = LinearRegression(learning_rate=0.1)
model.fit(X, y)

# Make predictions
X_test = np.array([[0], [1], [2]])
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")
print(f"Learned weights: {model.weights}")
print(f"Learned bias: {model.bias}")
```

### Example 2: Neural Network with Backpropagation
```python
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        # Hidden layer
        self.z1 = np.dot(self.W1, X.T) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2.T
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y.reshape(-1, 1)
        dW2 = (1/m) * np.dot(dz2.T, self.a1.T)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dz1 = np.dot(self.W2.T, dz2.T) * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(dz1, X)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
        
        # Update parameters
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = np.mean((y.reshape(-1, 1) - output)**2)
            
            # Backward pass
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test the neural network
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR problem

nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=1000)

# Test predictions
predictions = nn.forward(X)
print("XOR Predictions:")
for i, (x, true_y, pred_y) in enumerate(zip(X, y, predictions)):
    print(f"Input: {x}, True: {true_y}, Predicted: {pred_y[0]:.4f}")
```

### Example 3: Loss Function Derivatives
```python
def mse_loss_derivative(y_true, y_pred):
    """Derivative of MSE loss with respect to predictions"""
    return 2 * (y_pred - y_true)

def cross_entropy_loss_derivative(y_true, y_pred):
    """Derivative of cross-entropy loss with respect to predictions"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

# Example usage
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])

mse_grad = mse_loss_derivative(y_true, y_pred)
ce_grad = cross_entropy_loss_derivative(y_true, y_pred)

print("MSE Loss Gradients:", mse_grad)
print("Cross-Entropy Loss Gradients:", ce_grad)
```

## Key Takeaways

1. **Derivatives measure rate of change** - essential for optimization
2. **Chain rule is fundamental** - enables backpropagation in neural networks
3. **Partial derivatives** - needed for multi-variable functions
4. **Activation function derivatives** - crucial for gradient flow
5. **Loss function derivatives** - guide parameter updates

## Why Derivatives Matter in Neural Networks

1. **Gradient Descent**: Derivatives tell us which direction to adjust parameters
2. **Backpropagation**: Chain rule allows us to compute gradients through layers
3. **Optimization**: Derivatives help find minimum loss values
4. **Learning**: Derivatives quantify how much each parameter affects the output

## Next Steps

Now that you understand derivatives, you're ready to learn about:
- **Gradients**: Multi-dimensional derivatives
- **Backpropagation**: Using gradients to train neural networks
- **Optimization algorithms**: Advanced gradient-based methods

Derivatives are the mathematical foundation that makes neural network training possible. They tell us not just how functions change, but how to make them change in the direction we want!
