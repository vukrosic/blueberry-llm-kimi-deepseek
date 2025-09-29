# 02: The GLM-4 MoE Architecture

With a refreshed understanding of Mixture of Experts, let's now delve into the specific design choices and architectural details that characterize the MoE implementation in models like GLM-4. While the core principles remain the same, the devil is often in the details, and specific optimizations can significantly impact performance and training stability.

## Key Characteristics of GLM-4 Style MoE

GLM-4, like many other high-performing MoE models, typically employs a sparse MoE layer within the Feed-Forward Network (FFN) part of its Transformer blocks. Here are some common characteristics and design patterns:

1.  **Placement within the Transformer Block**: The MoE layer usually replaces the dense FFN in some or all of the Transformer blocks. This means that after the Multi-Head Attention sub-layer, the output passes through the MoE layer, followed by the usual residual connection and layer normalization.

2.  **Number of Experts (`N_experts`)**: GLM-4 models often feature a substantial number of experts, ranging from tens to hundreds (e.g., 64, 128, 256 experts). A larger number of experts contributes to higher model capacity.

3.  **Experts per Token (`Top-K`)**: For each token, the router typically selects a small, fixed number of top experts (e.g., K=2 or K=4). This `Top-K` selection is crucial for maintaining computational efficiency, as only these selected experts process the token.

4.  **Expert Architecture**: Each individual expert is usually a standard Feed-Forward Network (FFN) itself. This often means a two-layer MLP with an activation function (e.g., GELU, SwiGLU). The hidden dimension of these expert FFNs might be smaller than a dense FFN to manage memory, or it might be scaled to match the overall model capacity.

5.  **Gating Mechanism (Router)**:
    *   The router is typically a simple linear layer that projects the token's representation to `N_experts` logits.
    *   A `softmax` function is applied to these logits to get probabilities for each expert.
    *   The `Top-K` experts are then selected based on these probabilities.
    *   The output of the selected experts is weighted by their respective router probabilities before being summed.

    `Router(x) = Softmax(Linear(x))`

6.  **Load Balancing Loss**: To prevent a few experts from becoming overloaded while others remain underutilized, GLM-4 (and similar MoE models) incorporate an auxiliary **load balancing loss** during training. This loss term encourages the router to distribute tokens more evenly across all experts. It typically penalizes situations where experts are chosen too frequently or too rarely.

    *   This loss is added to the main language modeling loss during backpropagation.
    *   It helps ensure that all experts learn useful representations and contribute to the model's overall performance.

7.  **Expert Parallelism**: Due to the large number of experts, training and inference often involve **expert parallelism**. This means different experts are distributed across different GPUs or devices. When a token is routed to an expert on another device, its data is sent to that device for processing, and the result is sent back. This is a complex engineering challenge but essential for scaling MoE models.

## Conceptual Diagram of GLM-4 MoE Layer

```
Input Token Embedding (x)
      |
      v
+-----------------+
|     Router      |
| (Linear + Softmax)|
+--------^--------+
         | (Probabilities for N_experts)
         |
         | Select Top-K Experts
         v
+-------------------------------------------------------------------+
|  Expert 1  |  Expert 2  | ... |  Expert K  | ... | Expert N_experts |
|   (FFN)    |   (FFN)    |     |   (FFN)    |     |      (FFN)       |
+-------------------------------------------------------------------+
         ^ (Token routed to selected experts)
         |
         v
+-----------------+
| Weighted Sum of |
| Expert Outputs  |
+--------^--------+
         |
         v
      Output
```

Understanding these architectural choices is key to appreciating how GLM-4 achieves its impressive capabilities. In the next lesson, we will implement a GLM-4 style MoE layer in PyTorch, incorporating these design principles.
