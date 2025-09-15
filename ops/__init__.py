"""
GPU-adaptive operations package.

This package contains GPU-architecture-specific implementations of common
operations that automatically dispatch to the most optimized kernel available.
"""

from .matmul import matmul, matmul_with_info, get_available_kernels, print_kernel_info

__all__ = [
    'matmul',
    'matmul_with_info', 
    'get_available_kernels',
    'print_kernel_info',
]
