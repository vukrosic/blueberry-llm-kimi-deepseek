# Lesson 3: The Expert

So, what exactly is an "expert" in a Mixture of Experts layer?

It's surprisingly simple: **an expert is just a standard Feed-Forward Network (FFN)**.

## Just an FFN

That's right. Each expert in an MoE layer has the exact same architecture as the FFN we described in the first lesson of this section.

It's a two-layer Multi-Layer Perceptron:

1.  A linear layer that expands the input dimension.
2.  A non-linear activation function (like ReLU, GeLU, or SwiGLU).
3.  A linear layer that projects the dimension back down.

```
Expert(x) = (W2 * activation(W1 * x + b1)) + b2
```

## A Collection of FFNs

An MoE layer contains multiple of these experts. For example, an MoE layer might have 8 experts. This means it has 8 separate FFNs, each with its own set of weights (`W1`, `b1`, `W2`, `b2`).

```
Expert1(x) = (W2_1 * activation(W1_1 * x + b1_1)) + b2_1
Expert2(x) = (W2_2 * activation(W1_2 * x + b1_2)) + b2_2
...
Expert8(x) = (W2_8 * activation(W1_8 * x + b1_8)) + b2_8
```

During training, each of these experts will learn to specialize in different types of inputs, based on the data they are shown by the gating network.

## The Key Difference

The crucial difference between a standard FFN and an MoE layer is that **not all experts are used for every input**.

The gating network (which we'll cover next) selects which one or two experts are most relevant for the current input token. The other experts are not used, and their weights are not involved in the computation for that token.

This is what allows MoE models to be so large while remaining computationally manageable. You might have 64 experts, but for any given token, you might only use the two best ones.

In the next lesson, we'll look at the gating network, which is the component that makes this selection.