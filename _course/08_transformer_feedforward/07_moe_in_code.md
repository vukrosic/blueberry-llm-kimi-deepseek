# Lesson 7: MoE in Code, Step-by-Step

Let's build a Mixture of Experts (MoE) layer from the ground up using Python and NumPy. This will make the flow of data and the logic of the gating mechanism clear.

*Note: This implementation is for educational purposes. Real-world MoE models use highly optimized routing for efficiency on GPUs.*

---

### Part 1: The Expert

First, let's define our `Expert`. It's just a simple Feed-Forward Network (FFN) with one hidden layer.

```python
import numpy as np

class Expert:
    def __init__(self, input_dim, hidden_dim):
        # Each expert has its own weights
        self.w1 = np.random.randn(input_dim, hidden_dim)
        self.w2 = np.random.randn(hidden_dim, input_dim)

    def forward(self, x):
        # A simple FFN with a ReLU activation
        hidden = np.maximum(0, np.dot(x, self.w1))
        return np.dot(hidden, self.w2)

# --- Let's test it ---
input_dim = 4
hidden_dim = 8
expert1 = Expert(input_dim, hidden_dim)

# Create a single token embedding
token_input = np.random.randn(1, input_dim)

# Get the expert's output
expert_output = expert1.forward(token_input)
print("Shape of single expert output:", expert_output.shape)
```

---

### Part 2: The Gating Network

Next, let's see how the gating network produces the weights for each expert. It's just a linear layer followed by a softmax.

```python
# (Continuing from Part 1)

num_experts = 8

# The gating network is just one weight matrix
gate_weights = np.random.randn(input_dim, num_experts)

# Calculate the logits
gating_logits = np.dot(token_input, gate_weights)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# Get the probabilities for each expert
expert_probabilities = softmax(gating_logits)

print("\nGating probabilities for each expert:")
print(expert_probabilities)
print("\nSum of probabilities:", np.sum(expert_probabilities))
```
These probabilities tell us how much the gate "thinks" each expert should contribute to the final output for this specific token.

---

### Part 3: Top-k Selection (Simplified to k=1)

To save computation, we only want to use the expert(s) with the highest probability. Let's simplify this to `k=1` and find the single best expert.

```python
# (Continuing from Part 2)

# Find the index of the expert with the highest score
best_expert_index = np.argmax(expert_probabilities, axis=1)[0]

print(f"\nThe best expert for this token is: Expert #{best_expert_index}")
```
Now we know which expert to send our token to.

---

### Part 4: Combining the Output

The final step is to get the output from our chosen expert and weight it by its probability score.

```python
# (Continuing from Part 3)

# Create a list of our experts
experts = [Expert(input_dim, hidden_dim) for _ in range(num_experts)]

# Select the best expert
selected_expert = experts[best_expert_index]

# Get the output from only that expert
final_token_output = selected_expert.forward(token_input)

# In a top-k > 1 scenario, you would weight this output by its score
# For top-1, the weight is effectively 1, but we show it for clarity
final_token_output *= expert_probabilities[0, best_expert_index]

print("\nFinal output for the token:")
print(final_token_output)
```
We have successfully routed our input to a single expert and computed the output, saving the computation of the other 7 experts.

---

### Part 5: Putting It All Together (The MoELayer)

Now, let's wrap all this logic into a single `MoELayer` class that can handle a batch of tokens.

```python
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

class Expert:
    # (Paste the Expert class from Part 1 here)
    def __init__(self, input_dim, hidden_dim):
        self.w1 = np.random.randn(input_dim, hidden_dim)
        self.w2 = np.random.randn(hidden_dim, input_dim)

    def forward(self, x):
        hidden = np.maximum(0, np.dot(x, self.w1))
        return np.dot(hidden, self.w2)

class MoELayer:
    def __init__(self, input_dim, hidden_dim, num_experts):
        self.num_experts = num_experts
        self.experts = [Expert(input_dim, hidden_dim) for _ in range(num_experts)]
        self.gate = np.random.randn(input_dim, num_experts)

    def forward(self, x):
        # 1. Get gating probabilities
        gating_logits = np.dot(x, self.gate)
        gating_weights = softmax(gating_logits)

        # 2. Select the best expert for each token (top-1)
        best_expert_indices = np.argmax(gating_weights, axis=1)

        # 3. Compute final output (loop for clarity)
        final_output = np.zeros_like(x)
        for i, token_input in enumerate(x):
            expert_idx = best_expert_indices[i]
            selected_expert = self.experts[expert_idx]
            expert_output = selected_expert.forward(token_input)
            final_output[i] = gating_weights[i, expert_idx] * expert_output
        
        return final_output

# --- Example Usage ---
# A batch of 4 tokens, each with dimension 10
input_data = np.random.randn(4, 10)

moe_layer = MoELayer(input_dim=10, hidden_dim=20, num_experts=8)
output = moe_layer.forward(input_data)

print("\n--- Final MoE Layer Output ---")
print("Output shape:", output.shape)
print("Final output for the batch:\n", output)
```
This complete class shows how a batch of tokens can be efficiently processed, with each token being routed to its own specialized expert. This is the core principle that allows MoE models to scale to trillions of parameters.

---

**Next Lesson**: [The DeepSeek MLP](08_the_deepseek_mlp.md)