# Lesson 6: Attention in Code, Step-by-Step

Let's implement the self-attention mechanism piece by piece using Python and NumPy. This will connect the theory directly to the code.

---

### Part 1: Setup and Initialization

First, we need our input data (a sequence of word embeddings) and the weight matrices that will be used to create the Q, K, and V vectors.

```python
import numpy as np

# Input: 3 words, each with a 4-dimensional embedding
seq_len = 3
d_model = 4
embeddings = np.array([
    [1, 0, 1, 0],  # Word 1
    [0, 1, 0, 1],  # Word 2
    [1, 1, 0, 0]   # Word 3
])

# The dimension for our Q, K, and V vectors
d_k = 3 

# Weight matrices (initialized randomly)
W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_k)

print("Embeddings shape:", embeddings.shape)
print("W_Q shape:", W_Q.shape)
print("W_K shape:", W_K.shape)
print("W_V shape:", W_V.shape)
```

---

### Part 2: Creating Q, K, and V

Next, we create the Query, Key, and Value matrices by multiplying our embeddings with the weight matrices.

```python
# (Continuing from Part 1)

# Create Q, K, V matrices
Q = np.dot(embeddings, W_Q)
K = np.dot(embeddings, W_K)
V = np.dot(embeddings, W_V)

print("\nQ matrix shape:", Q.shape)
print("K matrix shape:", K.shape)
print("V matrix shape:", V.shape)
```
Each row in these new matrices corresponds to the Q, K, or V vector for a word.

---

### Part 3: Calculating Attention Scores

Now, we calculate the raw attention scores. This is done by taking the dot product of the Query matrix with the transpose of the Key matrix. This measures the similarity between each query and every key.

```python
# (Continuing from Part 2)

# Calculate raw scores
scores = np.dot(Q, K.T)

# Scale the scores to stabilize training
scaled_scores = scores / np.sqrt(d_k)

print("\nRaw scores shape:", scores.shape)
print("Scaled scores:")
print(scaled_scores)
```
The resulting matrix has a shape of `(seq_len, seq_len)`, where `scores[i, j]` is the attention score from word `i` to word `j`.

---

### Part 4: Applying Softmax

To turn our scores into a useful probability distribution, we apply the softmax function along each row.

```python
# (Continuing from Part 3)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)

print("\nAttention weights shape:", attention_weights.shape)
print("Attention weights (each row sums to 1):")
print(attention_weights)
```
These are our final attention weights. `attention_weights[i, j]` tells us how much attention word `i` should pay to word `j`.

---

### Part 5: Producing the Output

The final step is to create the new, context-aware representation for each word. We do this by multiplying our attention weights by the Value matrix.

```python
# (Continuing from Part 4)

# Multiply attention weights by V matrix
output = np.dot(attention_weights, V)

print("\nOutput shape:", output.shape)
print("Final output of the attention layer:")
print(output)
```
Each row in this `output` matrix is the new representation for the corresponding word, which is a blend of all other words' values, weighted by the attention scores.

---

### Putting It All Together

Here is a single function that encapsulates all the steps we just took.

```python
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def self_attention(embeddings, d_k):
    d_model = embeddings.shape[1]
    
    # 1. Initialize weights and create Q, K, V
    W_Q = np.random.randn(d_model, d_k)
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_k)
    Q = np.dot(embeddings, W_Q)
    K = np.dot(embeddings, W_K)
    V = np.dot(embeddings, W_V)
    
    # 2. Calculate and scale scores
    scores = np.dot(Q, K.T)
    scaled_scores = scores / np.sqrt(d_k)
    
    # 3. Softmax to get weights
    attention_weights = softmax(scaled_scores)
    
    # 4. Multiply weights by V
    output = np.dot(attention_weights, V)
    
    return output, attention_weights

# --- Example Usage ---
embeddings = np.array([[1,0,1,0], [0,1,0,1], [1,1,0,0]])
final_output, weights = self_attention(embeddings, d_k=3)

print("\n--- Final Function Output ---")
print("Final output:\n", final_output)
print("\nAttention weights:\n", weights)
```
This step-by-step process is the core of the self-attention mechanism and a fundamental building block of Transformers.