# Lesson 5: Multi-Head Attention

Self-attention is powerful, but what if we want the model to pay attention to different aspects of the input? For example, in the sentence "The cat sat on the mat", one part of the model might want to attend to the relationship between "cat" and "sat", while another part might want to attend to "cat" and "mat".

This is where **Multi-Head Attention** comes in.

## The Idea: Multiple Attention "Heads"

The idea is simple: instead of having just one set of Query, Key, and Value weight matrices (`W_Q`, `W_K`, `W_V`), we have multiple sets.

Each set is called an **attention head**. Each head can learn to focus on a different type of relationship in the data.

For example:

-   **Head 1** might learn to attend to subject-verb relationships.
-   **Head 2** might learn to attend to pronoun-referent relationships.
-   **Head 3** might learn to attend to local, sequential relationships.

## How it Works

1.  **Create Multiple Q, K, V Sets:** For each word, we create a separate set of Q, K, and V vectors for each attention head. If we have 8 heads, we will have 8 sets of `W_Q`, `W_K`, and `W_V` matrices.

2.  **Calculate Attention Independently:** Each attention head performs the attention calculation independently. This means each head calculates its own attention scores and produces its own output vector `z`.

    - `z_head1 = Attention(Q1, K1, V1)`
    - `z_head2 = Attention(Q2, K2, V2)`
    - ...and so on.

3.  **Concatenate and Project:** After all the heads have produced their output vectors, we concatenate them together.

    `z_concat = [z_head1, z_head2, ...]`

    This concatenated vector is then passed through another linear layer (with a weight matrix `W_O`) to produce the final output of the multi-head attention layer.

    `output = W_O * z_concat`

## Why Does This Work?

By having multiple heads, we give the model more representational power. Each head can specialize in a different aspect of the language.

It's like having a team of specialists. Instead of one generalist trying to understand everything, you have a team where each member focuses on their area of expertise. The final decision is then made by combining the insights from all the specialists.

In the original Transformer paper, they used 8 attention heads. This has become a common practice.

In the final lesson of this section, we'll look at how to implement a simple version of self-attention in Python.