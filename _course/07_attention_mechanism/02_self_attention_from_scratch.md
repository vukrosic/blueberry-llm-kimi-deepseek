# Lesson 2: Self-Attention From Scratch

Now let's get into the mechanics of self-attention. The key to understanding it is to understand the roles of three special vectors: **Query**, **Key**, and **Value**.

Imagine you're searching for a video on YouTube:

-   You type in a **Query** (e.g., "how to make pasta").
-   YouTube matches your query against a set of **Keys** (video titles, descriptions, etc.).
-   The videos with the best-matching keys are returned to you as **Values** (the videos themselves).

Self-attention works in a similar way, but it all happens within the model itself.

## From Words to Vectors

First, we take our input sentence and convert each word into an embedding vector. Let's say we have the sentence "The cat sat". We would have three embedding vectors, one for each word.

For each of these embedding vectors, we are going to create a Query, a Key, and a Value vector.

## Creating Q, K, and V

How do we create these vectors? We use three weight matrices, which are learned during training:

-   `W_Q`: A weight matrix to create Query vectors.
-   `W_K`: A weight matrix to create Key vectors.
-   `W_V`: A weight matrix to create Value vectors.

For a single word embedding `x`, we get the Q, K, and V vectors by multiplying `x` by each of these matrices:

-   `q = W_Q * x`
-   `k = W_K * x`
-   `v = W_V * x`

We do this for every word in our input sequence. So, for the sentence "The cat sat", we would have:

-   `q1`, `k1`, `v1` (for "The")
-   `q2`, `k2`, `v2` (for "cat")
-   `q3`, `k3`, `v3` (for "sat")

## The Intuition

-   **Query (q):** Represents the current word's "question" or what it's looking for. For example, the query for a verb might be looking for a subject.

-   **Key (k):** Represents what information a word is offering. The key for a noun might say, "I am a potential subject."

-   **Value (v):** Represents the actual content of the word. This is what gets passed on to the next layer if a word is attended to.

By comparing a word's Query vector with the Key vectors of all other words, the model can determine how much attention to pay to each of them.

In the next lesson, we'll walk through the exact calculation of attention scores using these Q, K, and V vectors, with real numbers.

---

**Next Lesson**: [Calculating Attention Scores](03_calculating_attention_scores.md)