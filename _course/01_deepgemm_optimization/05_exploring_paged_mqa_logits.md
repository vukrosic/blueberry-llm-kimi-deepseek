# Lesson 5: Exploring Paged MQA Logits

In our final deep dive into DeepGEMM's kernels, we'll explore the most advanced attention mechanism covered: Paged MQA logits. This technique is at the forefront of research for serving large language models (LLMs) efficiently, and DeepGEMM provides a cutting-edge implementation.

## Learning Objectives
- Understand the concept of Paged Attention and why it's important for LLM inference.
- Learn how Paged Attention solves the problem of memory fragmentation in the K/V cache.
- Analyze the `fp8_paged_mqa_logits` function in DeepGEMM.
- See how the metadata for Paged Attention is calculated.

## 1. The Challenge of Long Sequences: Memory Fragmentation

As we discussed in the previous lesson, the K/V cache is a major consumer of memory during LLM inference. When dealing with long sequences and large batches, managing this memory becomes a significant challenge. A common problem is **memory fragmentation**. Different sequences in a batch may have very different lengths, leading to inefficiently used memory blocks in the K/V cache.

## 2. Paged Attention to the Rescue

**Paged Attention** is a technique inspired by virtual memory and paging in operating systems. It works by dividing the K/V cache into fixed-size "blocks." Each sequence is then assigned a set of these blocks, which don't have to be contiguous in memory. A "block table" keeps track of which blocks belong to which sequence.

This approach has several advantages:

- **No Memory Fragmentation:** Since all blocks are the same size, there is no wasted memory.
- **Efficient Memory Sharing:**  For techniques like parallel decoding, where multiple output sequences are generated from the same prompt, the blocks for the prompt can be shared, saving a significant amount of memory.

## 3. The DeepGEMM Paged MQA Logits Kernel

DeepGEMM implements Paged Attention for MQA in its `fp8_paged_mqa_logits` kernel. This kernel is even more complex than the standard MQA kernel, as it has to handle the block table and the non-contiguous memory layout of the K/V cache.

Let's look at the function signature:

```cpp
static torch::Tensor fp8_paged_mqa_logits(const torch::Tensor& q,
                                          const torch::Tensor& fused_kv_cache,
                                          const torch::Tensor& weights,
                                          const torch::Tensor& context_lens,
                                          const torch::Tensor& block_table,
                                          const torch::Tensor& schedule_meta,
                                          const int& max_context_len,
                                          const bool& clean_logits) {
    // ...
}
```

- **`fused_kv_cache`**: This tensor contains the K/V cache stored in blocks.
- **`block_table`**: This tensor maps sequences to blocks in the K/V cache.
- **`schedule_meta`**: This is a metadata tensor that helps to schedule the work on the GPU efficiently.

### 3.1. Calculating the Metadata

Before calling the main kernel, a helper function `get_paged_mqa_logits_metadata` is used to create the `schedule_meta` tensor. This function analyzes the context lengths of the sequences in the batch and creates a schedule that balances the workload across the streaming multiprocessors (SMs) of the GPU.

```cpp
static torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int block_kv, int num_sms) {
    // ...
}
```

## 4. Practice Exercises

### Exercise 1: The Block Table
- Imagine you have a K/V cache with 10 blocks. You have two sequences in a batch. Sequence 1 needs 3 blocks, and Sequence 2 needs 4 blocks. Create a possible block table for this scenario.

### Exercise 2: The `ref_fp8_paged_mqa_logits` function
- The `tests/test_attention.py` file contains a reference implementation called `ref_fp8_paged_mqa_logits`. How does this function use the `block_tables` to access the K/V cache?

## AI Learning Prompt

> "I'm learning about Paged Attention for LLM inference. Can you explain the following concepts?
> 1. How does Paged Attention work, and how is it analogous to virtual memory in an operating system?
> 2. What is memory fragmentation, and why is it a problem for LLM inference?
> 3. How does Paged Attention enable memory sharing between different sequences?"

## Key Takeaways
- Paged Attention is a powerful technique for managing the K/V cache in LLM inference.
- It solves the problem of memory fragmentation and enables efficient memory sharing.
- DeepGEMM provides a highly optimized kernel for Paged MQA logits, which is essential for serving modern LLMs.

## Next Steps

Congratulations! You've now explored the core components of the DeepGEMM library. In the final lesson, we'll summarize what you've learned and discuss how to put it all together.

**Next Lesson**: [Conclusion and Putting It All Together](06_conclusion.md)
