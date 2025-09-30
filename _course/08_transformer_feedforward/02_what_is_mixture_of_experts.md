# Lesson 2: What is a Mixture of Experts?

The standard Feed-Forward Network (FFN) we saw in the last lesson is great, but it has a drawback: every part of the network is activated for every single input. This can be computationally expensive, especially as models get larger.

What if we could have a much larger network, but only use a small part of it for each input? This is the core idea behind the **Mixture of Experts (MoE)**.

## The Analogy: A Team of Specialists

Imagine you have a complex problem. Instead of giving it to one generalist, you could give it to a team of specialists. You would first have a "router" or a "manager" who looks at the problem and decides which specialist is best suited to solve it.

An MoE layer works in the same way:

1.  **Experts:** You have a set of "expert" networks. Each expert is a standard FFN, just like the one we saw in the previous lesson.

2.  **Gating Network (The Router):** You have a small "gating" network that looks at the input and decides which expert(s) to send it to.

This is a form of **conditional computation**. Instead of all the experts processing the input, only the selected ones do. This means you can have a very large number of experts (and thus a model with a huge number of parameters), but the actual computation for each input is much lower.

## Why is this Useful?

1.  **Parameter Efficiency:** You can have a model with trillions of parameters, but only a fraction of them are used for any given token. This allows for much larger and more capable models without a proportional increase in computational cost.

2.  **Specialization:** Each expert can learn to specialize in a different type of data or pattern. For example, one expert might become good at processing natural language, while another might specialize in code.

3.  **Faster Training (in theory):** Because you're only using a subset of the network for each input, training can be faster.

## Sparse vs. Dense Models

-   A standard Transformer with a regular FFN is a **dense** model. All parameters are used for all inputs.
-   A Transformer with an MoE layer is a **sparse** model. Only a fraction of the parameters are used for each input.

This shift from dense to sparse architectures is a key trend in modern large language models.

In the next lessons, we will break down the components of an MoE layer: the expert and the gating network.

---

**Next Lesson**: [The Expert](03_the_expert.md)