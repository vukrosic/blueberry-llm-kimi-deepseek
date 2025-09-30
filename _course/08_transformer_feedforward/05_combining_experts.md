# Lesson 5: Combining the Experts

We have our input, we have our experts, and our gating network has chosen which experts to use and assigned them weights. Now it's time to put it all together.

## The Final Output

The final output of the Mixture of Experts (MoE) layer is a **weighted sum of the outputs of the experts**.

The formula is simple:

`Output = sum(gating_weight_i * Expert_i(input))`

Where:
- `gating_weight_i` is the weight assigned to expert `i` by the gating network.
- `Expert_i(input)` is the output of expert `i` when it processes the input.

## An Example

Let's continue with our example from the previous lesson. We have 4 experts, and our gating network has selected the top 2:

-   **Expert 3:** weight = 0.28
-   **Expert 4:** weight = 0.27

(After re-normalizing these two weights so they sum to 1):
-   **Expert 3 weight (g3):** `0.28 / (0.28 + 0.27) = 0.51`
-   **Expert 4 weight (g4):** `0.27 / (0.28 + 0.27) = 0.49`

The weights for Expert 1 and Expert 2 are 0.

Now, we do the following:

1.  **Process the input with the selected experts:**
    -   `output_expert3 = Expert3(x)`
    -   `output_expert4 = Expert4(x)`

    (The other experts, Expert1 and Expert2, are not run at all. Their computation is skipped.)

2.  **Calculate the weighted sum:**
    -   `weighted_output3 = g3 * output_expert3`
    -   `weighted_output4 = g4 * output_expert4`

3.  **Combine:**
    -   `final_output = weighted_output3 + weighted_output4`

This `final_output` is the result of the MoE layer for our input token `x`.

## The Full Picture

So, for each token that flows through the Transformer block:

1.  It goes through the self-attention layer to get a context-aware representation.
2.  This representation is fed into the MoE layer.
3.  The gating network inside the MoE layer looks at the token and decides which 2 experts (or `k` experts) are the best fit.
4.  The token is processed by only those 2 experts.
5.  The outputs of those 2 experts are combined based on the gating weights.
6.  This combined output is the final result of the Transformer block.

This process allows the model to access a huge amount of knowledge (stored in the many experts) but use it in a very targeted and efficient way.

In the next lessons, we'll see how this fits into the larger Transformer architecture and look at a conceptual code implementation.

---

**Next Lesson**: [MoE in a Transformer](06_moe_in_a_transformer.md)