# Lesson 2: Understanding the DeepGEMM Codebase

Now that you have a high-level understanding of what DeepGEMM is, let's get our hands dirty by exploring its codebase. This lesson will walk you through the key files and directories, giving you a map to navigate the project.

## Learning Objectives
- Identify the main source code directories in DeepGEMM.
- Understand the purpose of the C++ source files (`.hpp`, `.cu`, `.cuh`).
- Learn where the Python bindings are defined.
- Discover how the custom kernels are tested.

## 1. The `csrc` Directory: The Heart of the Kernels

The `csrc` directory contains the C++ and CUDA source code for the high-performance kernels. This is where the low-level magic happens.

- **`csrc/apis/attention.hpp`**: This file defines the C++ API for the attention-related kernels. It's the bridge between the CUDA implementation and the PyTorch C++ extension.

- **`csrc/indexing/main.cu`**: This file seems to be the main entry point for the CUDA kernels, including implementations for various GEMM and attention operations.

- **`csrc/jit_kernels/impls/`**: This directory contains the implementations of the Just-In-Time (JIT) compiled kernels. You'll find files like:
    - `smxx_clean_logits.hpp`: A kernel to clean up logits (the output of a neural network layer before the activation function).
    - `smxx_fp8_mqa_logits.hpp`: The implementation of the FP8 Multi-Query Attention logits kernel.
    - `smxx_fp8_paged_mqa_logits.hpp`: The implementation of the FP8 Paged Multi-Query Attention logits kernel.

## 2. The `deep_gemm` Python Package

While the heavy lifting is done in C++ and CUDA, the project is ultimately used from Python. The `deep_gemm` directory contains the Python package.

- **`deep_gemm/__init__.py`**: This is the main file of the Python package. It imports the C++ extension and exposes the custom kernels to Python. You'll see lines like:

```python
from ._deep_gemm import (
    # ...
    fp8_mqa_logits,
    get_paged_mqa_logits_metadata,
    fp8_paged_mqa_logits,
    # ...
)
```

- **`deep_gemm/include/deep_gemm/impls/`**: This directory contains the CUDA header files (`.cuh`) for the kernel implementations. These files are included by the `.cu` files in `csrc`.
    - `sm90_fp8_mqa_logits.cuh`: The header for the FP8 MQA logits kernel for SM90 (Hopper) architecture.
    - `sm90_fp8_paged_mqa_logits.cuh`: The header for the FP8 Paged MQA logits kernel for SM90.
    - `smxx_clean_logits.cuh`: The header for the logit cleaning kernel.

## 3. The `tests` Directory: Ensuring Correctness

No software project is complete without tests! The `tests` directory contains Python scripts to test the custom kernels.

- **`tests/test_attention.py`**: This file contains unit tests for the attention-related kernels. It compares the output of the custom kernels with a reference implementation to ensure they are producing the correct results.

## 4. Practice Exercises

### Exercise 1: Find the Python API
- Open the `deep_gemm/__init__.py` file. Find the function signature for `fp8_mqa_logits`. What are its arguments?

### Exercise 2: Locate the Kernel Implementation
- In which file would you expect to find the main CUDA implementation for the `sm90_fp8_mqa_logits` kernel? Why?

## AI Learning Prompt

> "I'm exploring the codebase of a deep learning acceleration library called DeepGEMM. Can you explain the roles of the following file types in a PyTorch C++ extension?
> 1. `.hpp` files in a `csrc/apis` directory.
> 2. `.cu` and `.cuh` files.
> 3. `__init__.py` in a Python package that uses a C++ extension.
> 4. How does the code in a `.cu` file get called from Python?"

## Key Takeaways
- The core logic of DeepGEMM is in C++ and CUDA (`csrc` and `deep_gemm/include`).
- The Python API is defined in `deep_gemm/__init__.py`.
- Tests are crucial for verifying the correctness of the custom kernels.

## Next Steps

In the next lesson, we'll take a closer look at the FP8 GEMM kernels and start to understand how they work under the hood.

**Next Lesson**: [A Deep Dive into FP8 GEMM Kernels](03_deep_dive_into_fp8_gemm.md)
