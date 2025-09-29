# 02: A Deep Dive into Rotary Positional Embedding (RoPE)

In the previous lesson, we noted that Transformers need a way to understand the order of tokens. The most intuitive approach might be to have a big lookup table where each position (0, 1, 2, ...) has its own unique "positional vector" that we add to the token embedding. This is called an *absolute positional embedding*.

However, this method has drawbacks:
*   **Limited Length**: It only works for sequences up to the maximum length you defined for your lookup table. What if you want to analyze a longer document?
*   **Lack of Relative Sense**: It doesn't inherently teach the model the concept of "relative" position (e.g., that the distance between position 7 and 9 is the same as between 101 and 103).

**Rotary Positional Embedding (RoPE)** is a clever and elegant solution that addresses these issues. Instead of *adding* positional information, RoPE *rotates* the embedding vectors based on their position.

## The Core Idea: Encoding Position via Rotation

Imagine you have a vector. To encode its position, you could rotate it by a certain angle. If it's the first token, you rotate it by 5 degrees. If it's the second, you rotate it by 10 degrees, and so on.

The key insight of RoPE is that in self-attention, what truly matters is the dot-product between the **Query** vector (from one token) and the **Key** vector (from another token). If we rotate both the Query and the Key vectors by an amount corresponding to their absolute positions, the resulting dot-product will only depend on their *relative* positions.

## A Quick Math Detour: 2D Rotation Matrices

To understand RoPE, we first need to understand how to rotate a 2D vector. A vector `v = [x, y]` can be rotated counter-clockwise by an angle `θ` by multiplying it with a special **rotation matrix**.

The rotation matrix `R_θ` is defined as:

```
      [ cos(θ)  -sin(θ) ]
R_θ = [ sin(θ)   cos(θ) ]
```

So, the new rotated vector `v'` would be:

`v' = R_θ * v`

`[x'] = [ cos(θ)  -sin(θ) ] * [x]`
`[y'] = [ sin(θ)   cos(θ) ]   [y]`

This gives `x' = x*cos(θ) - y*sin(θ)` and `y' = y*cos(θ) + x*sin(θ)`. This is the fundamental operation at the heart of RoPE.

## From 2D to High Dimensions

Our token embeddings are not 2D; they are high-dimensional (e.g., 512, 4096, or more). How do we rotate a 512-dimensional vector?

RoPE's clever strategy is to **not** perform a complex high-dimensional rotation. Instead, it groups the dimensions of the embedding vector into pairs: `(d_0, d_1)`, `(d_2, d_3)`, ..., `(d_510, d_511)`.

It then rotates each pair independently as if it were a 2D vector!

However, each pair is rotated by a *different* angle `θ_i`.

## Defining the Rotation Angle `θ`

The angle of rotation must be a function of the token's position `m`. But we can't use the same angle for all the pairs. RoPE defines the rotation angle for the `i`-th pair of dimensions as:

`θ_{m,i} = m / (10000^(2i / d))`

Where:
*   `m` is the absolute position of the token (0, 1, 2, ...).
*   `d` is the total dimension of the embedding (e.g., 512).
*   `i` is the index of the dimension pair (from 0 to `d/2 - 1`).

This formula looks complex, but the intuition is simple:
*   The term `10000^(2i / d)` is the **wavelength**. It's a large number for small `i` and gets closer to 1 for large `i`.
*   This means the rotation speed (the "frequency") is very slow for the first pairs of dimensions and much faster for the later pairs.

Think of it like encoding a number using the hands of multiple clocks, all ticking at different speeds. The unique position of all the hands together represents the number. Here, the unique set of rotations represents the token's position.

## The Magic: How RoPE Encodes Relative Position

Now for the crucial part. Let's say we have a Query vector `q` at position `m` and a Key vector `k` at position `n`.

1.  We apply the rotary transformation to both:
    *   `q_m` = `R(m, i)` * `q`
    *   `k_n` = `R(n, i)` * `k`

2.  The attention score is based on their dot product, `<q_m, k_n>`.

Let's look at just one pair of dimensions for simplicity. Using complex numbers (which is a compact way to represent 2D rotation), the transformation is equivalent to `q_m = q * e^(i*m*θ)`.

The dot product `<q_m, k_n>` becomes:
`(q * e^(i*m*θ)) * (k * e^(i*n*θ))` (conjugate transpose)
`= q * k_conjugate * e^(i*m*θ) * e^(-i*n*θ)`
`= (q * k_conjugate) * e^(i*(m-n)*θ)`

Notice the result! The dot product between the two rotated vectors is equivalent to the dot product of the original vectors multiplied by a rotation that **only depends on the difference `m-n`**.

The absolute positions `m` and `n` are gone, and only their relative distance `m-n` remains. This is how the model learns relational information. The attention score between "the" (position 5) and "cat" (position 6) can be calculated in a way that is aware "cat" is `+1` token away.

## Implementation Sketch

In practice, we don't build the rotation matrices explicitly. We can directly calculate the `x'` and `y'` values.

```python
# q is a vector of shape (seq_len, dim)
# 1. Create the frequency term (theta) for each dimension pair
# theta = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))

# 2. Create the position index
# pos = torch.arange(seq_len)

# 3. Create the full angle matrix for each position and dimension
# freqs = torch.outer(pos, theta)
# angles = torch.cat((freqs, freqs), dim=-1) # Duplicate for each pair

# 4. Calculate sin and cos
# cos_vals = torch.cos(angles)
# sin_vals = torch.sin(angles)

# 5. Apply the rotation
# q_rotated = (q * cos_vals) + (rotate_half(q) * sin_vals)

# def rotate_half(x):
#     # Swaps and negates halves of the vector to simulate the [-y, x] part of rotation
#     x1, x2 = x.chunk(2, dim=-1)
#     return torch.cat((-x2, x1), dim=-1)
```

This pre-computes the `sin` and `cos` values for all positions up to a certain length and then applies them directly to the Q and K vectors during the attention calculation.

## Summary of Advantages

*   **Relative Positions**: It naturally encodes relative positional information, which is highly effective for attention mechanisms.
*   **Unbounded Length**: Because it doesn't rely on a fixed lookup table, RoPE can handle sequences of any length without needing to be retrained (though models have other practical limits).
*   **Good Performance**: It has become a standard and is used in many state-of-the-art LLMs like Llama and PaLM.

We have now established a robust method for the model to understand sequence order. In the next lesson, we will see how this fits into the main processing unit of our model: the Transformer Block.
