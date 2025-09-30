# Lesson 3: A Deep Dive into FP8 GEMM Kernels

In this lesson, we'll dive into the heart of DeepGEMM: the FP8 GEMM kernels. We'll explore the concepts behind FP8, the specialized instructions used, and how the code is structured to achieve maximum performance on NVIDIA's Hopper architecture.

## Learning Objectives
- Understand the motivation for using FP8 in deep learning.
- Learn about the key components of a CUDA kernel for GEMM.
- Get introduced to advanced concepts like TMA (Tensor Memory Accelerator) and WGMMA (Warpgroup Matrix-Multiply-Accumulate).
- Analyze snippets of the DeepGEMM FP8 MQA logits kernel.

## 1. Why FP8? The Need for Speed and Efficiency

As we discussed in Lesson 1, GEMM is a cornerstone of deep learning. Traditional models use 32-bit (FP32) or 16-bit (FP16/BF16) floating-point numbers. FP8, or 8-bit floating-point, is a more recent innovation that offers significant advantages:

- **Reduced Memory Usage:** FP8 numbers use half the memory of FP16/BF16, which means you can fit larger models and batches into GPU memory.
- **Faster Computation:**  Modern GPUs like the NVIDIA H100 have specialized hardware (Tensor Cores) that can perform FP8 matrix multiplications at a much higher rate than FP16 or FP32.

However, using FP8 is not without its challenges. The lower precision means there's a risk of losing accuracy. DeepGEMM's kernels are carefully designed to use FP8 where possible while minimizing the impact on model performance.

## 2. Anatomy of a DeepGEMM Kernel

Let's dissect the `sm90_fp8_mqa_logits.cuh` file to understand how a DeepGEMM kernel is built. These kernels are complex, so we'll focus on the high-level structure.

### 2.1. Kernel Signature

The kernel is defined as a C++ template function:

```cpp
template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_fp8_mqa_logits(...) {
    // ...
}
```

- **Templates:** The use of templates allows the compiler to generate specialized versions of the kernel for different configurations (e.g., number of attention heads, head dimension).
- **`__global__`:** This keyword indicates that this is a CUDA kernel that can be launched from the host (CPU) and will run on the device (GPU).
- **`__launch_bounds__`:** This is a performance tuning hint that tells the compiler about the expected launch configuration.

### 2.2. TMA: High-Speed Data Transfer

Inside the kernel, you'll see references to TMA, which stands for **Tensor Memory Accelerator**. This is a new feature in the Hopper architecture that provides a very efficient way to load data from global GPU memory into the much faster shared memory of a streaming multiprocessor (SM).

```cpp
// Prefetch TMA descriptors
if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    // ...
}

// ...

// Issue TMA Q
tma_copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
```

The code first prefetches the TMA descriptors and then uses `tma_copy` to initiate the data transfer.

### 2.3. WGMMA: The Matrix Multiplication Powerhouse

WGMMA stands for **Warpgroup Matrix-Multiply-Accumulate**. This is the instruction that performs the actual matrix multiplication on the Tensor Cores. The `wgmma` instruction operates on small tiles of matrices that are loaded into registers.

```cpp
// Issue WGMMA
#pragma unroll
for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
    auto desc_a = make_smem_desc(smem_kv[kv_stage_idx] + (warpgroup_idx * WGMMA::M) * kHeadDim + k * WGMMA::K,
                                 to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
    auto desc_b = make_smem_desc(smem_q[q_stage_idx] + k * WGMMA::K,
                                 to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
    WGMMA::wgmma(desc_a, desc_b, accum, k);
}
```

The code iterates in a loop, calling `WGMMA::wgmma` to multiply tiles of the input matrices and accumulate the results.

## 3. Practice Exercises

### Exercise 1: Find the WGMMA instruction
- Look at the `sm90_fp8_mqa_logits.cuh` file in the `DeepGEMM_course_materials.txt`. Can you find the line where the `WGMMA::wgmma` instruction is called?

### Exercise 2: The Role of Shared Memory
- The code uses shared memory (`smem_q`, `smem_kv`, etc.). Why is shared memory so important for the performance of this kernel?

## AI Learning Prompt

> "I'm studying a CUDA kernel for FP8 GEMM from the DeepGEMM project. Can you explain these concepts in more detail?
> 1. What is the relationship between a streaming multiprocessor (SM), a warp, and a warpgroup on an NVIDIA GPU?
> 2. What is TMA (Tensor Memory Accelerator) and how does it differ from regular global memory loads?
> 3. What is a WGMMA (Warpgroup Matrix-Multiply-Accumulate) instruction and how does it use Tensor Cores?"

## Key Takeaways
- FP8 offers a significant speedup but requires careful implementation.
- DeepGEMM kernels are complex C++ templates that are highly specialized for the hardware.
- TMA and WGMMA are key Hopper architecture features that are essential for achieving high performance in GEMM operations.

## Next Steps

Now that you've had a glimpse into the FP8 GEMM kernels, we'll move on to another important optimization in DeepGEMM: Multi-Query Attention.

**Next Lesson**: [Understanding Multi-Query Attention (MQA) Logits](04_understanding_mqa_logits.md)
