# Lesson 4: Understanding Multi-Query Attention (MQA) Logits

In this lesson, we shift our focus from the GEMM operation itself to a specific application where it's used: attention mechanisms in transformer models. We'll explore Multi-Query Attention (MQA), a memory-saving variant of the standard multi-head attention, and how DeepGEMM provides a highly optimized kernel for it.

## Learning Objectives
- Understand the difference between Multi-Head Attention (MHA) and Multi-Query Attention (MQA).
- Learn how the MQA logits kernel in DeepGEMM is designed.
- Analyze the `fp8_mqa_logits` function and its parameters.
- Appreciate the performance benefits of a fused MQA logits kernel.

## 1. From Multi-Head to Multi-Query Attention

Transformer models, the backbone of modern NLP, rely on a mechanism called **Multi-Head Attention (MHA)**. In MHA, the model has multiple "attention heads," each with its own set of query (Q), key (K), and value (V) projections. This allows the model to attend to different parts of the input sequence simultaneously.

However, MHA can be memory-intensive, especially during inference, because the K and V caches for all heads need to be stored in GPU memory. This is where **Multi-Query Attention (MQA)** comes in. In MQA, all attention heads share a single K and V projection. This dramatically reduces the size of the K/V cache, saving memory and bandwidth.

## 2. The DeepGEMM MQA Logits Kernel

DeepGEMM includes a specialized kernel, `fp8_mqa_logits`, designed specifically for MQA. This kernel is "fused," which means it combines multiple operations (in this case, the QK^T matrix multiplication and the subsequent reduction) into a single CUDA kernel. This fusion has several performance benefits:

- **Reduced Memory Traffic:** By performing the operations in a single pass, the kernel avoids writing intermediate results to and reading them from global memory.
- **Improved Parallelism:** The kernel is designed to maximize the utilization of the GPU's compute resources.

Let's look at the function signature from `csrc/apis/attention.hpp`:

```cpp
static torch::Tensor fp8_mqa_logits(const torch::Tensor& q,
                                    const std::pair<torch::Tensor, torch::Tensor>& kv,
                                    const torch::Tensor& weights,
                                    const torch::Tensor& cu_seq_len_k_start,
                                    const torch::Tensor& cu_seq_len_k_end,
                                    const bool& clean_logits) {
    // ...
}
```

- **`q`**: The query tensor.
- **`kv`**: A pair of tensors representing the key and value caches.
- **`weights`**: The attention weights.
- **`cu_seq_len_k_start` and `cu_seq_len_k_end`**: These tensors define the start and end positions for each sequence in the batch, which is important for handling variable-length sequences.
- **`clean_logits`**: A boolean flag to indicate whether to clean up the logits outside the valid attention mask.

## 3. Testing the MQA Logits Kernel

The `tests/test_attention.py` file provides a great example of how to use and test the `fp8_mqa_logits` kernel. The test function `test_mqa_logits` does the following:

1.  Generates random input tensors for Q, K, and weights.
2.  Calls the `deep_gemm.fp8_mqa_logits` function.
3.  Compares the output with a reference implementation (`ref_fp8_mqa_logits`) to verify correctness.
4.  Benchmarks the performance of the kernel.

```python
def test_mqa_logits():
    print('Testing FP8 MQA Logits:')
    num_heads, head_dim = 64, 128
    for seq_len in (2048, 4096):
        for seq_len_kv in (4096, 8192, 16384, 32768, 65536, 131072):
            # ...
            logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
            # ...
```

This test demonstrates the wide range of sequence lengths that the kernel is optimized for.

## 4. Practice Exercises

### Exercise 1: MQA vs. MHA
- Draw a diagram that illustrates the difference between Multi-Head Attention and Multi-Query Attention in terms of the K and V projections.

### Exercise 2: The `ref_fp8_mqa_logits` function
- Look at the `ref_fp8_mqa_logits` function in `tests/test_attention.py`. How does it implement the attention mechanism? How does it differ from the fused kernel?

## AI Learning Prompt

> "I'm learning about attention mechanisms in transformer models. Can you explain the following?
> 1. What is the difference between Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA)?
> 2. What are the pros and cons of using MQA instead of MHA?
> 3. What is a "fused kernel" in the context of CUDA programming, and why is it beneficial for performance?"

## Key Takeaways
- MQA is a memory-efficient alternative to MHA.
- DeepGEMM provides a fused kernel for MQA logits that is highly optimized for performance.
- The `fp8_mqa_logits` kernel is a key component for accelerating inference of modern transformer models.

## Next Steps

In the next lesson, we'll look at an even more advanced attention mechanism: Paged MQA logits, which is designed for handling very long sequences.

**Next Lesson**: [Exploring Paged MQA Logits](05_exploring_paged_mqa_logits.md)
