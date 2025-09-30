# Neuron From Scratch: Building a Neuron in Python

We've learned about the linear step and the activation step. Now it's time to combine them into a single, reusable `Neuron` class in Python.

## 1. The Goal: Encapsulate the Neuron's Logic

We want to create a Python class that represents a single neuron. This class will:
1.  Be initialized with the number of inputs it should expect.
2.  Have its own internal `weights` and `bias` that are randomly initialized.
3.  Have a `forward` method that takes inputs and performs the full calculation: `activation( (weights · inputs) + bias )`.

## 2. Designing the `Neuron` Class

Let's think about the structure.

```python
import torch

class Neuron:
    def __init__(self, num_inputs):
        # The neuron needs to know how many inputs to expect.
        # This determines the size of the weight vector.
        self.num_inputs = num_inputs

        # Initialize weights randomly and bias at zero.
        # This is a common starting point.
        self.weights = torch.randn(num_inputs)
        self.bias = torch.zeros(1) # Or just 0.0

    def forward(self, inputs):
        # This method performs the neuron's main calculation.
        # 1. The linear step
        score = torch.dot(self.weights, inputs) + self.bias

        # 2. The activation step
        # We'll use Sigmoid for this example.
        activation = torch.sigmoid(score)

        return activation
```

This class design neatly packages all the logic for a single neuron.

## 3. Using Our Neuron

Let's create an instance of our `Neuron` and pass some data through it.

```python
# Create a neuron that accepts 3 inputs
neuron = Neuron(num_inputs=3)

# Let's inspect its initial random weights and bias
print(f"Initial weights: {neuron.weights}")
print(f"Initial bias: {neuron.bias}")

# Create some example input data
inputs = torch.tensor([0.5, 0.1, 0.8])

# Get the neuron's output for these inputs
output = neuron.forward(inputs)

print(f"\nInputs: {inputs}")
print(f"Output: {output}")
```

Every time you run this, you'll get a different output because the weights are initialized randomly. This randomness is crucial for the learning process to begin.

## 4. Why Use a Class?

Using a class is a powerful concept in programming that maps perfectly to deep learning:
- **State**: The class holds the neuron's "state" – its `weights` and `bias`. These are the parameters that will be updated during training.
- **Behavior**: The class defines the neuron's "behavior" through its `forward` method.
- **Reusability**: We can now create as many neurons as we want from this single blueprint.

## 5. Practice Examples

### Practice 1: Create a Bigger Neuron

Instantiate a neuron that is designed to take 10 inputs. Create a random tensor of 10 inputs and pass it through your neuron's `forward` method.

### Practice 2: Change the Activation Function

Modify your `Neuron` class. Add an `activation_function` parameter to the `__init__` method. Let the user pass in a function like `torch.sigmoid` or `torch.relu`.

```python
class FlexibleNeuron:
    def __init__(self, num_inputs, activation_function):
        self.num_inputs = num_inputs
        self.weights = torch.randn(num_inputs)
        self.bias = torch.zeros(1)
        # Store the function itself
        self.activation = activation_function

    def forward(self, inputs):
        score = torch.dot(self.weights, inputs) + self.bias
        # Call the stored function
        return self.activation(score)

# Now you can create different kinds of neurons!
sigmoid_neuron = FlexibleNeuron(num_inputs=2, activation_function=torch.sigmoid)
relu_neuron = FlexibleNeuron(num_inputs=2, activation_function=torch.relu)
```
Try to complete this `FlexibleNeuron` and test it with some sample data.

---
Our neuron can now take inputs and produce an output. But how do we know if this output is any good? In the next lesson, we'll learn how to make a prediction and compare it to a desired outcome.

---

**Next Lesson**: [Making a Prediction](05_making_a_prediction.md)

```
