# 05: The Final Output Layer: From Vector to Vocabulary

We've journeyed through the bulk of the Transformer. Our input tokens have been embedded, their positions encoded with RoPE, and they have been repeatedly refined by a stack of Transformer Blocks. The result is a sequence of final, context-rich hidden state vectors—one for each input token.

Now we arrive at the final, crucial step: turning this internal representation into a concrete prediction for the next token in the sequence.

This process involves two main components: a **Linear Layer** to create scores for every word in our vocabulary, and a **Softmax Function** to turn those scores into probabilities.

## The Goal: Predicting the Next Token

For a task like language generation, we are trying to predict the token that should follow our input sequence. For example, if the input is "The cat sat on the", we want the model to predict "mat".

To do this, we only need to look at the final hidden state vector corresponding to the *last* token of our input (the vector for "the"). This vector is the most informed, as it has gathered context from all the tokens that came before it through the self-attention mechanism.

This final vector, with a dimension of `d_model` (e.g., 512), is the culmination of the model's understanding. Our job is to project this single vector into the space of our entire vocabulary.

## The Final Linear Layer (The "Un-embedding")

This projection is done by a single, standard linear layer (`torch.nn.Linear`). This layer acts as a bridge from the model's internal dimension to the vocabulary dimension.

*   **Input Dimension**: `d_model` (e.g., 512)
*   **Output Dimension**: `vocab_size` (e.g., 50,257)

The weight matrix of this linear layer has a shape of `[d_model, vocab_size]`. When we pass our final hidden state vector (shape `[1, d_model]`) through it, we get a new vector of shape `[1, vocab_size]`. This large output vector contains a raw score, or **logit**, for every single token in our vocabulary.

### A Clever Trick: Weight Tying

Here's an elegant and efficient optimization used in many modern Transformers. Remember the very first layer of our model, the **Embedding Layer**? It contains a weight matrix of shape `[vocab_size, d_model]` that maps token IDs to vectors.

The final linear layer has a weight matrix of shape `[d_model, vocab_size]`. Notice that this is just the transpose of the embedding matrix!

**Weight Tying** is the practice of making these two layers share the same weight matrix. The final linear layer simply uses the *transpose* of the embedding layer's weights.

**Why do this?**
1.  **Parameter Efficiency**: This trick almost halves the number of parameters in a large model, as the embedding and final layers are often two of the largest.
2.  **Improved Performance**: It's based on the intuition that the mapping *from* a word to a vector and the mapping *from* a vector back *to* a word should be consistent. If the model knows how to represent "cat" as a vector, it should use that same knowledge to recognize that vector as representing "cat". This shared representation has been shown to improve model quality.

## From Logits to Probabilities: The Softmax Function

The output of our final linear layer is a vector of logits. These are raw, unbounded numbers. A logit of `5.2` for the token "mat" and `-1.3` for "house" clearly indicates the model prefers "mat", but these are not probabilities. They don't sum to 1, and they aren't confined to a `[0, 1]` range.

This is where the **Softmax function** comes in. It's a mathematical operation that takes a vector of arbitrary real numbers and transforms it into a valid probability distribution.

Here's how it works for a logit vector `L = [l_1, l_2, ..., l_vocab_size]`:

1.  **Exponentiate**: It takes the mathematical constant `e` (approx. 2.718) and raises it to the power of each logit: `[e^l_1, e^l_2, ...]`. This has two effects:
    *   It makes all the scores positive.
    *   It exaggerates the differences. A logit that is slightly larger than another becomes exponentially larger.

2.  **Sum**: It sums up all these exponentiated values to get a single number, the normalization constant: `Sum = e^l_1 + e^l_2 + ...`

3.  **Divide**: It divides each individual exponentiated value by this sum. The probability `p_i` for the `i`-th token is:

    `p_i = (e^l_i) / Sum`

The result is a new vector of the same size, where:
*   Every value is between 0 and 1.
*   The sum of all values is exactly 1.

We now have a clean probability distribution. The token "mat" might have a probability of `0.92`, "rug" might have `0.03`, and all other tokens have very small probabilities that sum to `0.05`.

## Making a Prediction

Now that we have probabilities, we can finally choose our next token.

*   **Greedy Search (Argmax)**: The most straightforward approach is to simply find the token with the highest probability (using `argmax`) and select it. While simple, this can lead to repetitive and deterministic text.

*   **Sampling**: A more common approach is to treat the probability distribution as a weighted lottery and *sample* from it. If "mat" has a 92% probability, it will be chosen 92% of the time, but occasionally, the model might choose "rug", introducing a degree of randomness that makes the generated text feel more natural and creative.

This sampling can be controlled with techniques like **temperature scaling** (to make the distribution more or less random) and **top-k/top-p sampling** (to avoid sampling from the long tail of very unlikely tokens).

We have now completed the full forward pass of a Transformer, from input tokens to a final, usable prediction. The only remaining step is to put all these components together in code.

---

**Next Lesson**: [Full Transformer in Code](06_full_transformer_in_code.md)
