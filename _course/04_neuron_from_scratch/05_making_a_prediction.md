# Neuron From Scratch: Making a Prediction

Our neuron can take inputs and produce an output. We call this process of passing inputs through the neuron to get an output the **forward pass**. The output itself is our neuron's **prediction**.

## 1. The Goal: Interpret the Neuron's Output

The output of our Sigmoid neuron is a number between 0 and 1. How do we turn this into a concrete prediction?

We can set a **threshold**. For a binary (Yes/No) decision, a natural threshold is 0.5.
- If the output is > 0.5, we predict "Yes" (or `1`).
- If the output is <= 0.5, we predict "No" (or `0`).

## 2. A Concrete Example: Exam Pass/Fail

Let's imagine our neuron is trying to predict whether a student will pass an exam. The inputs are:
1.  Hours studied (normalized)
2.  Hours slept (normalized)

Let's create our neuron and make a prediction.

```python
import torch

# We'll reuse the Neuron class from the previous lesson
class Neuron:
    def __init__(self, num_inputs):
        self.weights = torch.randn(num_inputs)
        self.bias = torch.zeros(1)

    def forward(self, inputs):
        score = torch.dot(self.weights, inputs) + self.bias
        return torch.sigmoid(score)

# --- The Prediction Task ---

# 1. Create a neuron for our 2-input problem
pass_fail_neuron = Neuron(num_inputs=2)

# Let's manually set some weights and bias to make it predictable
# These are not learned yet! We're just assigning them.
pass_fail_neuron.weights = torch.tensor([5.0, 2.0]) # Strong weight on studying
pass_fail_neuron.bias = torch.tensor([-8.0])      # A high negative bias (hard to pass)

# 2. Create input data for a student
# Student 1: Studied a lot (1.0), slept okay (0.8)
student_1_inputs = torch.tensor([1.0, 0.8])

# 3. Get the neuron's output (the probability)
output_prob = pass_fail_neuron.forward(student_1_inputs)

# 4. Make a final prediction based on the threshold
prediction = 1 if output_prob > 0.5 else 0

print(f"--- Student 1 ---")
print(f"Inputs: {student_1_inputs}")
print(f"Output Probability: {output_prob.item():.4f}")
print(f"Final Prediction: {'Pass' if prediction == 1 else 'Fail'}")
```

Let's trace the calculation:
`Score = (1.0 * 5.0) + (0.8 * 2.0) + (-8.0)`
`Score = 5.0 + 1.6 - 8.0 = -1.4`
`Activation = sigmoid(-1.4) ≈ 0.19`
Since 0.19 is less than 0.5, the prediction is "Fail".

## 3. The Problem of Randomness

When we first create a neuron, its weights are **random**. This means its initial predictions are complete guesses. They will be wrong.

```python
# Create a new neuron with random weights
random_neuron = Neuron(num_inputs=2)

# Let's use the same student data
inputs = torch.tensor([1.0, 0.8])

# The output will be random
random_output = random_neuron.forward(inputs)
random_prediction = 1 if random_output > 0.5 else 0

print(f"\n--- Random Neuron ---")
print(f"Initial random weights: {random_neuron.weights}")
print(f"Output Probability: {random_output.item():.4f}")
print(f"Final Prediction: {'Pass' if random_prediction == 1 else 'Fail'}")
```
The prediction from this random neuron is meaningless.

## 4. The Core Question of Learning

So, we have a prediction. But we also have the **ground truth** (the actual outcome). For Student 1, let's say they actually **did pass** the exam (ground truth = 1).

- Our neuron predicted `0` (Fail).
- The truth is `1` (Pass).

Our neuron was **wrong**.

The fundamental question of machine learning is: **How do we measure *how wrong* the neuron was, and how do we use that information to adjust its weights and bias to make it less wrong next time?**

This is the concept of **loss** and **optimization**, which we'll explore next.

## 5. Practice Examples

### Practice 1: A Different Student

Using the `pass_fail_neuron` with the fixed weights (`[5.0, 2.0]`) and bias (`-8.0`):
- A new student studied very little (0.2) but slept a lot (1.0).
- Create the input tensor for this student.
- Pass it through the neuron and get the output probability.
- Determine the final prediction (Pass/Fail).

### Practice 2: The Power of Bias

What if we change the bias of the `pass_fail_neuron` to `0.0`? Rerun the prediction for Student 1. Does the prediction change? What does this show about the bias's role in setting the decision boundary?

---
We can make a prediction, but it's useless if we can't tell how good it is. Next, we'll learn how to quantify the neuron's error using a loss function.
