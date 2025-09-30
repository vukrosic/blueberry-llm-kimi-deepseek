# Lesson 4: Applying Attention Weights

We've calculated our attention weights. Now what? It's time to use them to create the final, context-aware representation of our word. This is the "information blending" step.

## The Blending Process

The process is simple: we take a weighted sum of all the **Value vectors** (`v1`, `v2`, `v3`), using the attention weights we just calculated.

Let's recall our values from the previous lesson:

**Attention Weights for "The":**
- Weight for `v1` ("The"): `0.39`
- Weight for `v2` ("cat"): `0.39`
- Weight for `v3` ("sat"): `0.22`

**Value Vectors:**
- `v1 = [0, 1, 1]`
- `v2 = [1, 0, 1]`
- `v3 = [1, 1, 0]`

Now, we multiply each Value vector by its corresponding attention weight:

- `0.39 * v1 = 0.39 * [0, 1, 1] = [0, 0.39, 0.39]`
- `0.39 * v2 = 0.39 * [1, 0, 1] = [0.39, 0, 0.39]`
- `0.22 * v3 = 0.22 * [1, 1, 0] = [0.22, 0.22, 0]`

Finally, we sum these weighted vectors together to get our output vector, `z1`:

`z1 = [0, 0.39, 0.39] + [0.39, 0, 0.39] + [0.22, 0.22, 0]`
`z1 = [0 + 0.39 + 0.22, 0.39 + 0 + 0.22, 0.39 + 0.39 + 0]`
`z1 = [0.61, 0.61, 0.78]`

This vector `z1` is the new, attention-enhanced representation for the word "The".

## What Just Happened?

Instead of just using the original embedding for "The", we have created a new representation that is a blend of "The", "cat", and "sat".

- It contains `39%` of the information from "The".
- It contains `39%` of the information from "cat".
- It contains `22%` of the information from "sat".

This new vector `z1` understands that "The" in this sentence is closely related to "cat". This is the power of self-attention! It allows the model to build representations of words that are deeply aware of the context they appear in.

We would repeat this exact same process for every other word in the sentence to get their attention-enhanced representations (`z2` for "cat" and `z3` for "sat").

## The Big Picture

The full calculation for the output of a self-attention layer can be summarized in a single, beautiful formula:

`Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V`

This is exactly what we just did, but expressed in matrix form for efficiency.

In the next lesson, we'll look at an extension of this idea called Multi-Head Attention, which allows the model to learn even richer representations.

---

**Next Lesson**: [Multi-Head Attention](05_multi_head_attention.md)