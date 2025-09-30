# 09: Tokenization and Embeddings

So far, we've focused on what happens once the data is already in the form of numbers (tensors). But how do we get there? Neural networks don't understand words or characters; they only understand numbers. This lesson covers the critical pre-processing step that turns raw text into meaningful vectors that a model can process.

This process involves two key ideas: **Tokenization** and **Embeddings**.

## 1. Tokenization: From Text to Integers

**Tokenization** is the process of breaking down a piece of text into smaller units called **tokens**. A token can be a word, a character, or a part of a word.

There are a few ways to do this:

*   **Word-based Tokenization**: This is the simplest approach. You split the text by spaces. "The cat sat" becomes `["The", "cat", "sat"]`. The problem is that your vocabulary (the set of all possible tokens) can become enormous. You also have no way to handle out-of-vocabulary (OOV) words like "perambulate" if it wasn't in your training data.

*   **Character-based Tokenization**: Here, you split the text into individual characters: `['T', 'h', 'e', ' ', 'c', 'a', 't', ...]` This keeps the vocabulary very small, but it creates very long sequences and loses the semantic meaning of a whole word.

*   **Subword Tokenization (The Modern Standard)**: This is the approach used by almost all modern LLMs (e.g., BPE, WordPiece). It's a hybrid approach. Common words like "cat" remain as single tokens, but rare or complex words are broken down into smaller, meaningful subwords. For example, "tokenization" might become `["token", "##ization"]`. This keeps the vocabulary size manageable while still being able to represent any word.

### The Vocabulary

After the tokenizer is trained on a massive corpus of text, it generates a **vocabulary**: a definitive list of all possible tokens it can produce. Each unique token is then mapped to a unique integer.

Our vocabulary might look something like this:
*   `"<unk>"`: 0 (for unknown tokens)
*   `"the"`: 1
*   `"a"`: 2
*   ...
*   `"cat"`: 543
*   `"##ization"`: 4872
*   ...

Now, we can take any text and convert it into a sequence of integers. "The cat" becomes `[1, 543]`.

## 2. Embeddings: From Integers to Vectors

We now have numbers, but the integers `1` and `543` have no inherent meaning or relationship to each other. We need to convert these integers into rich, dense vectors that can capture semantic meaning. This is the job of the **Embedding Layer**.

An embedding layer (`torch.nn.Embedding` in PyTorch) is essentially a big **lookup table** (a matrix).

*   The number of **rows** in this matrix is the size of our vocabulary (`vocab_size`).
*   The number of **columns** is the size of our desired embedding vector (`embedding_dimension` or `d_model`).

When you pass a token ID (like `543`) into the embedding layer, it simply returns the **543rd row** of the matrix. This row is the vector representation, or **embedding**, for the token "cat".

### Embeddings are Learnable

This is the most important part: the embedding matrix is not static. It is initialized with random numbers, and its values are **learnable parameters** of the model. During training, when the model makes an error, gradients are backpropagated all the way back to the embedding layer, and the vector for "cat" is nudged in a direction that helps the model make a better prediction next time. 

Over time, the model learns to place words with similar meanings close to each other in the vector space. The vector for "cat" will end up being very similar to the vector for "kitten" and somewhat similar to "dog", but very different from "car".

### Code Example

Let's see this in action.

```python
import torch
import torch.nn as nn

# 1. Define a small vocabulary size and embedding dimension
vocab_size = 1000  # Our vocabulary has 1000 unique tokens
embedding_dim = 128 # We want to represent each token with a 128-dimensional vector

# 2. Create the embedding layer
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# 3. Our tokenized input (a batch of 2 sequences)
# Let's pretend "The cat" is [1, 543] and "A dog" is [2, 678]
input_ids = torch.tensor([
    [1, 543],
    [2, 678]
], dtype=torch.long)

print(f"Shape of input IDs: {input_ids.shape}")

# 4. Pass the IDs through the embedding layer
output_embeddings = embedding_layer(input_ids)

print(f"Shape of output embeddings: {output_embeddings.shape}")
# The output shape will be (batch_size, sequence_length, embedding_dim)
# (2, 2, 128)

# Let's inspect the embedding for the first word of the first sequence ("The")
first_word_embedding = output_embeddings[0, 0, :]
print(f"\nEmbedding for token ID 1:\n {first_word_embedding}")

# This output vector is what gets fed into the first Transformer block.
```

This process—turning text into token IDs and then looking up those IDs in a learnable embedding matrix—is the starting point for all modern language models.
