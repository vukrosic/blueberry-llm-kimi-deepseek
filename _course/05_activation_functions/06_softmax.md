# Activation Functions: Softmax

Softmax is a unique activation function because it's not applied element-wise in the same way as the others. Instead, it's applied across a vector of scores and converts them into a probability distribution.

## 1. The Goal: Convert Scores to Probabilities

Softmax is used almost exclusively as the final activation function in a **multi-class classification** network.

Imagine a network trying to classify an image into one of three categories: "cat", "dog", or "bird". The final linear layer of the network will output a vector of raw scores (logits), one for each class.

`logits = [2.5, 1.8, 3.2]`  (score for cat, dog, bird)

Our goal is to convert this vector of arbitrary scores into a vector of probabilities that sum to 1.

## 2. What it Does

The Softmax function is defined as:
`Softmax(zᵢ) = eᶻᵢ / Σ(eᶻⱼ)`

In simple terms, for each score `zᵢ` in the vector `z`:
1.  Calculate `e` raised to the power of `zᵢ`. This makes all scores positive.
2.  Sum up all the exponentiated scores from step 1.
3.  Divide the result of step 1 by the result of step 2.

This process ensures two things:
- Every output value is between 0 and 1.
- The sum of all output values is exactly 1.

## 3. How to Use It

Let's apply Softmax to our example logits.

```python
import torch
import torch.nn.functional as F

# Raw output scores (logits) from a model for a single sample
# Scores for "cat", "dog", "bird"
logits = torch.tensor([2.5, 1.8, 3.2])

# Apply the softmax function
# We specify dim=-1 to apply it across the last dimension of the tensor
probabilities = F.softmax(logits, dim=-1)

print(f"Original logits:   {logits}")
print(f"Probabilities:     {probabilities}")
print(f"Sum of probs:      {torch.sum(probabilities)}")
print(f"Predicted class index: {torch.argmax(probabilities)}")
```

The output shows the model's confidence for each class. The highest probability corresponds to the "bird" class (index 2), so that would be the model's prediction.

## 4. Softmax vs. Sigmoid

- **Sigmoid**: Used for **binary** classification (one output neuron). The output is the probability of the positive class.
- **Softmax**: Used for **multi-class** classification (one output neuron *per class*). The outputs represent a probability distribution across all classes.

You can think of Sigmoid as a special case of Softmax where the number of classes is 2.

## 5. Numerical Stability: Logits and LogSoftmax

In practice, calculating `eᶻ` can lead to very large numbers, causing numerical instability (overflow). PyTorch has a more stable way of handling this by working in "log space".

- **LogSoftmax**: `F.log_softmax(logits, dim=-1)` calculates the log of the probabilities directly in a stable way.
- **Cross-Entropy Loss**: The standard loss function for multi-class classification, `torch.nn.CrossEntropyLoss`, is designed to take the *raw logits* as input. It internally performs a LogSoftmax and then calculates the loss.

**Best Practice**: For multi-class classification, your model should output the raw logits and you should use `torch.nn.CrossEntropyLoss` as your loss function. You only need to apply Softmax explicitly when you want to view the final probabilities for interpretation.

## 6. Practice Examples

### Practice 1: Temperature
The "temperature" of a Softmax function can be controlled by dividing the logits by a scalar `T` before applying the function: `F.softmax(logits / T, dim=-1)`.
- A high temperature (`T > 1`) makes the probabilities more uniform (less confident).
- A low temperature (`T < 1`) makes the probabilities more extreme (more confident).

Take the `logits` from our example and apply Softmax with `T=3.0` and `T=0.5`. What happens to the output distribution?

### Practice 2: Batch of Logits
Models make predictions for a whole batch of samples at once. Your logits tensor might have a shape of `(batch_size, num_classes)`, e.g., `(4, 3)`.

Create a random `(4, 3)` tensor of logits. Apply `F.softmax(dim=-1)`. Check the shape of the output. Then, verify that for each sample in the batch (each row), the probabilities sum to 1.

---
This concludes our series on activation functions! You've seen the classical functions, modern replacements, and the special functions used for classification outputs.
