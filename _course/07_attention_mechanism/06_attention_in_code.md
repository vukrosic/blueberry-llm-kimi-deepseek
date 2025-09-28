# Lesson 6: Attention in Code

Let's implement a simple version of self-attention using Python and NumPy. This will help solidify the concepts we've learned.

We'll implement a single attention head.

```python
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# 1. Input Data
# Let's say we have 3 words, and their embeddings are 4-dimensional.
# (seq_len, d_model)
embeddings = np.array([
    [1, 0, 1, 0],  # Word 1
    [0, 1, 0, 1],  # Word 2
    [1, 1, 0, 0]   # Word 3
])

# 2. Weight Matrices
# We need W_Q, W_K, W_V matrices to create Q, K, V vectors.
# Let's say our d_k (dimension of Q and K) is 3.
# (d_model, d_k)
d_model = 4
d_k = 3

W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_k) # d_v is often same as d_k

# 3. Create Q, K, V
# (seq_len, d_k)
Q = np.dot(embeddings, W_Q)
K = np.dot(embeddings, W_K)
V = np.dot(embeddings, W_V)

# 4. Calculate Attention Scores
# (seq_len, seq_len)
scores = np.dot(Q, K.T)

# 5. Scale
scaled_scores = scores / np.sqrt(d_k)

# 6. Softmax
attention_weights = softmax(scaled_scores)

print("Attention Weights:")
print(attention_weights)

# 7. Apply Attention Weights to V
# (seq_len, d_k)
output = np.dot(attention_weights, V)

print("\nOutput:")
print(output)

```

## What the Code is Doing

1.  **Input Data:** We start with our input embeddings for a sequence of 3 words.

2.  **Weight Matrices:** We initialize the weight matrices `W_Q`, `W_K`, and `W_V` with random values. In a real model, these are learned during training.

3.  **Create Q, K, V:** We perform matrix multiplication to get our `Q`, `K`, and `V` matrices.

4.  **Calculate Scores:** We get the raw attention scores by multiplying `Q` with the transpose of `K`.

5.  **Scale:** We scale the scores.

6.  **Softmax:** We apply softmax to get the final attention weights.

7.  **Apply to V:** We multiply the attention weights by the `V` matrix to get the final output.

Each row of the `output` matrix is the new, context-aware representation for the corresponding word in the input.

## Conclusion

You have now seen the attention mechanism from a high-level concept all the way down to a code implementation. This powerful technique is a fundamental building block of many state-of-the-art NLP models.

Next, we will explore the other key component of a Transformer block: the Feed-Forward Network.
