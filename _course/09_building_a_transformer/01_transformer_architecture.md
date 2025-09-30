# 01: The Transformer Architecture: A High-Level View

Before we dive into the code, let's zoom out and look at the blueprint of the Transformer model we're about to build. While the original Transformer had an "Encoder-Decoder" structure for machine translation, many modern language models (like GPT) use a "Decoder-Only" architecture. This is what we will focus on, as it is simpler and perfectly suited for language generation.

## The Core Components

Imagine the model as a series of processing stations. A piece of text (a sequence of tokens) enters at the bottom and flows upwards, being transformed at each step until it emerges as a prediction for the next word.

Here is the journey of our data:

1.  **Input & Embedding**:
    *   It all starts with a sequence of input tokens (e.g., `[101, 2054, 2003, 102]`).
    *   These numbers are fed into an **Embedding Layer**, which converts each token ID into a rich vector representation (e.g., a 512-dimensional vector). This vector is the model's internal "understanding" of the token.

2.  **Positional Encoding**:
    *   The Transformer architecture itself doesn't know the order of the tokens. It sees the input as a "bag" of vectors.
    *   To solve this, we inject **Positional Encoding** into the token embeddings. This gives the model crucial information about the sequence order. We will be using a sophisticated method called **Rotary Positional Embedding (RoPE)**.

3.  **A Stack of Transformer Blocks**:
    *   The core of the model is a stack of identical **Transformer Blocks** (e.g., 12, 24, or even 96 blocks stacked on top of each other).
    *   The positionally-encoded embeddings pass through these blocks one by one.
    *   Each block refines the vectors, allowing tokens to "talk" to each other and build a deeper contextual understanding.

4.  **Inside a Transformer Block**:
    *   **Multi-Head Self-Attention**: This is the first sub-layer. It's where tokens look at all other tokens in the sequence to figure out which ones are most important for understanding their own meaning.
    *   **Feed-Forward Network (FFN)**: This is the second sub-layer. After the attention mechanism gathers context, the FFN processes each token's vector individually. In our model, this will be a **Mixture of Experts (MoE)** layer, which is a more advanced and efficient type of FFN.

5.  **The Final Output Layer**:
    *   After passing through the entire stack of Transformer blocks, we have a final set of refined vectors—one for each input token.
    *   We only care about the vector for the *very last* token in the sequence, as this vector holds the most up-to-date information needed to predict the *next* token.
    *   This final vector is passed through a single **Linear Layer** (also called the "un-embedding" or "head" layer). This layer projects our internal model dimension (e.g., 512) up to the size of our entire vocabulary (e.g., 50,257).

6.  **Softmax**:
    *   The output of the linear layer is a raw vector of numbers called **logits**.
    *   The **Softmax** function is applied to this logit vector to convert it into a probability distribution. Each value now represents the model's confidence that a specific token in the vocabulary is the correct next token. We can then sample from this distribution to generate text.

## Visualizing the Flow

Here's a simplified diagram of the decoder-only architecture:

```
      +--------------------------+
      |      Next Token Probabilities     |
      +-------------^------------+
                    |
              +-----------+
              |  Softmax  |
              +-----------+
                    |
        +----------------------+
        | Final Linear Layer   |
        +----------------------+
                    | (Use output of last token)
      +--------------------------+
      |                          |
      |  Transformer Block N     |
      | (Self-Attention -> FFN)  |
      |                          |
      +-------------^------------+
                    .
                    .
                    .
      +-------------|------------+
      |                          |
      |  Transformer Block 1     |
      | (Self-Attention -> FFN)  |
      |                          |
      +-------------^------------+
                    |
+----------------------------------------+
| Token Embeddings + Positional Encoding |
+--------------------^-------------------+
                     |
              +--------------+
              | Input Tokens |
              +--------------+
```

In the next lessons, we will build each of these components, starting with the clever way Transformers handle sequence order: Rotary Positional Encoding.

---

**Next Lesson**: [RoPE Positional Encoding](02_rope_positional_encoding.md)
