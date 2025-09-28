# Lesson 3: Calculating Attention Scores

This is where we get to the numbers. Let's walk through the calculation of attention scores for a single word. We'll use the sentence "The cat sat".

Let's assume we have our Query, Key, and Value vectors for each word. For simplicity, let's say these vectors are 3-dimensional.

**Word 1 ("The"):**
- `q1 = [1, 0, 0]`
- `k1 = [1, 1, 0]`
- `v1 = [0, 1, 1]`

**Word 2 ("cat"):**
- `q2 = [0, 1, 0]`
- `k2 = [1, 1, 0]`
- `v2 = [1, 0, 1]`

**Word 3 ("sat"):**
- `q3 = [0, 0, 1]`
- `k3 = [0, 1, 1]`
- `v3 = [1, 1, 0]`

We want to calculate the attention-enhanced representation for the first word, "The".

### Step 1: Calculate Scores

We take the Query vector of "The" (`q1`) and compute the dot product with the Key vectors of all the words in the sentence (`k1`, `k2`, `k3`).

- **Score for "The" on "The":** `q1 . k1 = [1, 0, 0] . [1, 1, 0] = 1*1 + 0*1 + 0*0 = 1`
- **Score for "The" on "cat":** `q1 . k2 = [1, 0, 0] . [1, 1, 0] = 1*1 + 0*1 + 0*0 = 1`
- **Score for "The" on "sat":** `q1 . k3 = [1, 0, 0] . [0, 1, 1] = 1*0 + 0*1 + 0*1 = 0`

These scores represent how much the query for "The" matches the keys of the other words.

### Step 2: Scale

We then divide these scores by the square root of the dimension of the key vectors (`d_k`). In our case, `d_k = 3`. So we divide by `sqrt(3) ≈ 1.732`.

- `1 / 1.732 = 0.577`
- `1 / 1.732 = 0.577`
- `0 / 1.732 = 0`

This scaling step is important to prevent the dot products from becoming too large, which can cause issues with training.

### Step 3: Softmax

Next, we apply the softmax function to these scaled scores. Softmax converts the scores into a probability distribution, where all the values are between 0 and 1 and sum to 1.

`softmax([0.577, 0.577, 0])`

- `e^0.577 = 1.78`
- `e^0.577 = 1.78`
- `e^0 = 1`

Sum = `1.78 + 1.78 + 1 = 4.56`

- **Attention weight for "The":** `1.78 / 4.56 = 0.39`
- **Attention weight for "cat":** `1.78 / 4.56 = 0.39`
- **Attention weight for "sat":** `1 / 4.56 = 0.22`

These are our **attention weights**. They tell us that to understand the word "The" in this context, the model should pay about 39% of its attention to "The", 39% to "cat", and 22% to "sat".

In the next lesson, we'll see how to use these attention weights to create the final output vector.