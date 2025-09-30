# 03: Building a Transformer Block

Now that we have our token embeddings and a way to encode their positions with RoPE, we can move to the main processing unit of the language model: the **Transformer Block**.

The entire body of a Transformer is just a stack of these identical blocks. A 12-layer model has 12 of these blocks stacked on top of each other. The goal of each block is to take a sequence of vectors and refine them, producing a new sequence of vectors that has a deeper contextual understanding.

A Transformer Block has two main sub-layers:

1.  **Multi-Head Self-Attention**
2.  **A Feed-Forward Network** (in our case, a Mixture of Experts layer)

Crucially, these layers are connected by two other components: **Residual Connections** and **Layer Normalization**. Without these, the model would fail to train.

## The Data Flow

Here is the path the data takes through a single Transformer Block. We will use the "Pre-Norm" configuration, which is common in modern LLMs, as it tends to provide more stable training.

`Input: x`

1.  `norm_x = LayerNorm(x)`
2.  `attention_output = MultiHead_SelfAttention(norm_x)`
3.  `x = x + attention_output`  *(Residual Connection 1)*

4.  `norm_x = LayerNorm(x)`
5.  `ffn_output = FeedForward_Network(norm_x)`
6.  `x = x + ffn_output`  *(Residual Connection 2)*

`Output: x`

This output `x` is then fed as the input to the next Transformer Block in the stack.

![Transformer Block Diagram](https://i.imgur.com/l422tCF.png)

## A Deep Dive: Residual Connections and Layer Normalization

Why do we need these `+ x` and `LayerNorm` steps? Why not just connect the attention and FFN layers directly?

### The Problem: Vanishing & Exploding Gradients

When you stack many layers in a neural network, you can run into a serious problem during training. As the gradients are backpropagated from the final layer all the way down to the first layer, they are multiplied by the weights of each layer they pass through. 

*   If these weights are consistently small (less than 1), the gradients can shrink exponentially, becoming so tiny they are effectively zero (**Vanishing Gradients**). The early layers of the model stop learning.
*   If the weights are consistently large (greater than 1), the gradients can grow exponentially until they are enormous (**Exploding Gradients**). This makes the training process unstable.

### Solution 1: Residual Connections (The "Information Highway")

A residual connection, or skip connection, is incredibly simple: you just add the input of a layer to its output.

`output = input + Layer(input)`

This simple addition creates a direct "information highway" through the network. The gradient can now flow directly back through the `+` operation, completely bypassing the layer and its multiplications. This ensures that even in a very deep network, the gradients can reach the earliest layers without vanishing.

It also helps the model learn. Imagine a layer needs to learn an "identity" function (i.e., just pass the input through unchanged). Without a residual connection, the layer would have to learn to make its weights approximate an identity matrix, which is difficult. With a residual connection, the layer only needs to learn to output zero! The `input` is already being passed through, so the layer can easily "do nothing" if that's the best course of action.

### Solution 2: Layer Normalization

Layer Normalization is a technique to stabilize the network by controlling the values (the "activations") flowing through it. It works on a per-token basis.

For each token's vector in the sequence, LayerNorm does the following:
1.  Calculates the mean and variance of all the values *within that single vector*.
2.  Normalizes the vector by subtracting the mean and dividing by the standard deviation. Now the vector has a mean of 0 and a standard deviation of 1.
3.  It then scales and shifts this normalized vector with two learnable parameters, `gamma` (a weight) and `beta` (a bias). This allows the model to decide if and how much to undo the normalization if it's beneficial.

By ensuring that the inputs to the main layers (Attention and FFN) are always well-behaved (i.e., not too large or too small), LayerNorm prevents the activations from spiraling out of control and drastically improves the stability of training deep networks.

**Pre-Norm vs. Post-Norm**: In the original Transformer paper, the normalization was applied *after* the residual connection (Post-Norm). However, later research found that applying it *before* (Pre-Norm), as we have shown, leads to more stable training and often better performance, which is why it has become a popular choice.

## The Sub-Layers Revisited

*   **Multi-Head Self-Attention**: This is where the magic happens. As we've discussed, this layer allows each token to look at every other token in the sequence (including itself) and decide which ones are most important for building its own contextual representation. The RoPE transformations are applied to the Query and Key vectors *inside* this module.

*   **Feed-Forward Network (FFN)**: After the attention layer has gathered context from across the sequence, the FFN's job is to process this new information. It operates on each token's vector independently. In our model, we will use a **Mixture of Experts (MoE)** layer here, which is a more powerful and efficient type of FFN that we explored in Module 8.

With this block structure, we can now stack them to create a deep, powerful model. Each block adds another layer of processing and refinement, allowing the model to capture increasingly complex patterns in the data.

---

**Next Lesson**: [The Final Linear Layer](05_the_final_linear_layer.md)
