# Lesson 4: The Gate

The gating network (or "router") is the heart of the Mixture of Experts layer. It's the component that decides which experts to use for a given input.

## How it Works

The gating network is a very simple neural network itself. It's usually just a single linear layer followed by a softmax function.

1.  **Input:** The gating network takes the input token's representation (the output from the attention layer).

2.  **Linear Layer:** It multiplies the input by a weight matrix, `W_g`. The size of this matrix is `(input_dimension, num_experts)`.

    `logits = W_g * x`

    The output of this layer is a vector of "logits", one for each expert.

3.  **Softmax:** The softmax function is applied to these logits. This converts them into a probability distribution that sums to 1.

    `gating_weights = softmax(logits)`

    The resulting `gating_weights` vector contains the probability that each expert should be used for the current input.

## An Example with Numbers

Let's say we have 4 experts and our input `x` is a 3-dimensional vector.

-   `x = [1, 0, 0.5]`
-   The gating weight matrix `W_g` will have a shape of `(3, 4)`.

`W_g = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]`

1.  **Calculate Logits:**
    `logits = x @ W_g` (matrix multiplication)
    `logits = [1*0.1 + 0*0.5 + 0.5*0.9, ...]`
    `logits = [0.55, 0.7, 0.85, 1.0]`

2.  **Apply Softmax:**
    `softmax([0.55, 0.7, 0.85, 1.0])`

    This will produce a result like:
    `gating_weights = [0.21, 0.24, 0.28, 0.27]`

This means:
- Expert 1 gets a weight of 0.21
- Expert 2 gets a weight of 0.24
- Expert 3 gets a weight of 0.28
- Expert 4 gets a weight of 0.27

## Top-k Gating

In practice, using all the experts with these small weights is inefficient. Instead, most MoE models use **Top-k gating**, where they only select the `k` experts with the highest weights.

For example, with `k=2` in our example above, the gating network would select **Expert 3** and **Expert 4**.

The weights for these top-k experts are then re-normalized (so they sum to 1), and the weights for all other experts are set to 0.

This ensures that only a small, fixed number of experts are used for each token, which is the key to the computational savings of MoE.

In the next lesson, we'll see how to combine the outputs of these selected experts.