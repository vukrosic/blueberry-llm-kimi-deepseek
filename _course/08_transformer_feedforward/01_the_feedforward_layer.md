# Lesson 1: The Feed-Forward Layer

Inside a Transformer block, after the attention mechanism has done its job of gathering and blending information, the result is passed to a **Feed-Forward Network (FFN)**.

This FFN is a relatively simple component, but it plays a crucial role.

## What is it?

The FFN in a Transformer is a standard Multi-Layer Perceptron (MLP). It consists of two linear layers with a non-linear activation function in between.

For each word's representation coming out of the attention layer, the FFN processes it independently.

The process looks like this:

1.  **First Linear Layer:** This layer expands the dimensionality of the input. A common practice is to expand it by a factor of 4. For example, if the input dimension is 512, this layer would project it to a dimension of 2048.

    `output1 = W1 * x + b1`

2.  **Activation Function:** A non-linear function like ReLU (Rectified Linear Unit) is applied. This introduces non-linearity, which is essential for the model to learn more complex patterns.

    `output2 = ReLU(output1)`

3.  **Second Linear Layer:** This layer projects the result back down to the original input dimension (e.g., from 2048 back to 512).

    `output3 = W2 * output2 + b2`

This final output is the result of the FFN for that word.

## Why is it Important?

The attention layer is primarily responsible for understanding the relationships between words and gathering context. You can think of it as handling the "information routing" part.

The FFN, on the other hand, is responsible for the **transformation** of that information. It takes the context-rich representations from the attention layer and processes them, allowing the model to learn more complex features.

While the attention layers are linear transformations (just matrix multiplications), the non-linearity of the FFN is what gives the Transformer its depth and allows it to approximate very complex functions.

## The "Point-Wise" Nature

An important detail is that the FFN is applied to each word's representation **independently and identically**. The same weight matrices (`W1`, `W2`) are used for every word in the sequence.

This means that while the attention layer is busy mixing information between words, the FFN provides a consistent transformation for each word based on its new, context-aware representation.

In the next lessons, we will explore a more advanced and efficient version of the FFN called the Mixture of Experts (MoE).