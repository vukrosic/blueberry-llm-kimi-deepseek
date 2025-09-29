# 02: DeepSeek Latent Attention Architecture

Having understood the general concept of latent attention, let's now dive into the specific architectural details of how DeepSeek models implement this mechanism. DeepSeek's approach often involves a structured interaction between the input tokens and a set of learnable latent tokens, designed to efficiently capture global context.

## Key Components and Flow

DeepSeek's latent attention typically integrates into the Transformer block, replacing or augmenting the standard self-attention layer. The core idea revolves around a fixed, small number of **learnable latent vectors** that act as global memory or summary points.

Let's denote:
*   `X`: The input sequence of token embeddings (Query, Key, Value for standard attention).
*   `L`: A small, fixed set of learnable latent vectors (e.g., 64 or 128 latent tokens).

The process generally involves a few distinct attention steps:

1.  **Input-to-Latent Attention (Compression)**:
    *   **Purpose**: To compress the information from the entire input sequence `X` into the latent vectors `L`.
    *   **Mechanism**: Queries are derived from `L`, and Keys/Values are derived from `X`. Each latent vector attends to all input tokens. This allows the latent vectors to form a summary of the input sequence.
    *   `L_new = Attention(Q=L, K=X, V=X)`

2.  **Latent Self-Attention (Refinement)**:
    *   **Purpose**: To allow the latent vectors to interact with each other and refine their global summary. This is a standard self-attention operation performed only among the latent vectors.
    *   **Mechanism**: Queries, Keys, and Values are all derived from `L_new` (or `L`).
    *   `L_refined = SelfAttention(Q=L_new, K=L_new, V=L_new)`

3.  **Latent-to-Input Attention (Expansion/Retrieval)**:
    *   **Purpose**: To inject the refined global context from `L_refined` back into the original input tokens `X`.
    *   **Mechanism**: Queries are derived from `X`, and Keys/Values are derived from `L_refined`. Each input token attends to the refined latent vectors to retrieve relevant global information.
    *   `X_new = Attention(Q=X, K=L_refined, V=L_refined)`

This sequence ensures that the input tokens first contribute to a global summary, the summary is refined, and then the refined summary is used to enhance the representation of each input token.

## Architectural Details

*   **Learnable Latent Vectors**: The initial `L` vectors are typically initialized as learnable parameters (`nn.Parameter`) of the model. They are not derived from the input but are learned over the course of training.
*   **Number of Latent Tokens**: This is a hyperparameter, usually much smaller than the maximum sequence length (e.g., 64, 128, 256). A smaller number leads to greater compression and efficiency but might lose fine-grained details.
*   **Projection Layers**: As with standard attention, linear projection layers (`W_Q`, `W_K`, `W_V`, `W_O`) are used to transform the input `X` and latent `L` vectors into Query, Key, and Value representations for each attention step.
*   **Multi-Head Mechanism**: Each of these attention steps (Input-to-Latent, Latent Self-Attention, Latent-to-Input) is typically performed using a multi-head mechanism, allowing the model to capture different types of relationships simultaneously.
*   **Residual Connections and Layer Normalization**: These are applied around each attention sub-layer, just as in a standard Transformer block, to ensure stable training and effective information flow.

## Conceptual Diagram

```
Input Tokens (X)
      |
      v
+-----------------+
| Input-to-Latent |
|    Attention    |
+--------^--------+
         |
         v
  Latent Tokens (L_new)
         |
         v
+-----------------+
| Latent Self-    |
|    Attention    |
+--------^--------+
         |
         v
  Latent Tokens (L_refined)
         |
         v
+-----------------+
| Latent-to-Input |
|    Attention    |
+--------^--------+
         |
         v
Output Tokens (X_new)
```

This architecture allows DeepSeek models to efficiently process long sequences by distilling information into a compact latent space, refining that information, and then re-integrating it into the token representations. In the next lesson, we will implement this mechanism in PyTorch.
