# 01: Revisiting Mixture of Experts (MoE)

Before diving into the specifics of GLM-4's Mixture of Experts (MoE) implementation, let's quickly recap the core concepts of MoE layers. We first introduced MoE in Module 8 as an advanced form of the Feed-Forward Network (FFN) within a Transformer block.

## What is Mixture of Experts?

Traditional Transformer models use dense FFNs, meaning every input token passes through the *same* FFN. While effective, this approach means that as models grow larger, the computational cost and memory footprint increase proportionally.

**Mixture of Experts (MoE)** layers offer a solution by introducing **sparse activation** and **conditional computation**. Instead of every token activating every parameter in a large FFN, each token is routed to only a small subset of specialized FFNs, called **experts**.

## Why Use MoE?

1.  **Increased Model Capacity**: MoE allows for a massive increase in the total number of parameters in a model without a proportional increase in computational cost during inference. A model with 100 experts, where each token uses 2, effectively has 50 times more parameters than a dense model of the same size, but only uses 2x the computation.
2.  **Efficiency**: During inference, only a fraction of the total parameters are activated for any given token, leading to more efficient computation compared to a dense model of equivalent capacity.
3.  **Specialization**: Different experts can specialize in different types of data or tasks. For example, one expert might become good at processing code, another at scientific text, and another at creative writing.
4.  **Handling Diversity**: MoE layers are particularly effective for tasks with diverse inputs, where different parts of the input might benefit from different processing pathways.

## Core Components of an MoE Layer

An MoE layer typically consists of two main parts:

1.  **The Router (or Gate)**:
    *   This is a small neural network (often a simple linear layer followed by a softmax) that takes the input token's representation.
    *   Its job is to decide which experts are most relevant for processing this particular token.
    *   It outputs a probability distribution over all available experts.

2.  **The Experts**:
    *   These are typically independent Feed-Forward Networks (FFNs), each with its own set of parameters.
    *   Each expert is designed to process a specific type of information or handle a particular aspect of the data.

## How it Works (Conceptual Flow)

For each input token:

1.  The token's representation is fed into the **Router**.
2.  The Router outputs scores (or probabilities) for each expert. Based on these scores, the Router selects the **top-K** experts (e.g., K=2) that are most suitable for processing this token.
3.  The token's representation is then sent to these selected experts.
4.  Each selected expert processes the token independently.
5.  The outputs from the selected experts are then combined, usually by weighting them according to the scores provided by the Router, to produce the final output for that token.

### Load Balancing

A common challenge with MoE is **expert imbalance**, where a few experts become dominant and are chosen for most tokens, while others are underutilized. To mitigate this, a **load balancing loss** is often added during training. This loss encourages the router to distribute tokens more evenly across all experts, ensuring that all experts get a chance to learn and specialize.

This recap provides the necessary foundation. In the next lesson, we will examine how GLM-4 specifically designs and integrates its MoE layers into its Transformer architecture.
