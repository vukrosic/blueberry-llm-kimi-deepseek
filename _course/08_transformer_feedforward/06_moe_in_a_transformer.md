# Lesson 6: MoE in a Transformer

We've learned about all the components of a Mixture of Experts (MoE) layer. Now let's see how it fits into the overall architecture of a Transformer.

## Replacing the FFN

In a standard Transformer, each block contains two main sub-layers:

1.  A Multi-Head Self-Attention layer.
2.  A standard Feed-Forward Network (FFN).

In an MoE-based Transformer, the architecture is very similar. The only difference is that the FFN is **replaced by an MoE layer**.

So, a block in an MoE Transformer looks like this:

1.  A Multi-Head Self-Attention layer.
2.  A Mixture of Experts (MoE) layer.

This change is often only made in every other Transformer block. For example, in a 12-layer model, the odd-numbered layers might have a standard FFN, while the even-numbered layers have an MoE layer.

This is done to balance model performance and training stability.

## The Flow of Data

Let's trace the path of a single token through an MoE Transformer block:

1.  **Input:** The token's embedding enters the block.

2.  **Self-Attention:** The token attends to all other tokens in the sequence. The multi-head attention mechanism produces a context-aware representation of the token.

3.  **Residual Connection & Layer Norm:** The output of the attention layer is added to the original input (a residual connection), and the result is normalized.

4.  **MoE Layer:** This normalized representation is then fed into the MoE layer.
    -   The **gating network** selects the top-k experts.
    -   The token is processed by these `k` **experts**.
    -   The outputs of the experts are combined.

5.  **Residual Connection & Layer Norm:** The output of the MoE layer is added to its input (the output of the attention sub-layer), and the result is normalized.

6.  **Output:** This final vector is the output of the Transformer block for that token, which is then passed to the next block.

## The Big Advantage

By using MoE layers, a model can have a vastly larger number of parameters without a proportional increase in the amount of computation needed to process a token. This is the key to building massive models like GPT-4.

A model might have 1 trillion total parameters, but for any given token, it might only use 20 billion of them. This makes it possible to train and serve models that would otherwise be computationally infeasible.

In our final lesson, we'll look at a conceptual implementation of an MoE layer in Python.

---

**Next Lesson**: [MoE in Code](07_moe_in_code.md)