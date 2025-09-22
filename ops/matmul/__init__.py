"""
T4-optimized matmul operations.

This module provides simplified matrix multiplication operations
optimized specifically for Tesla T4 GPU.
"""

import torch
from typing import Tuple
from system import SYSTEM_CONFIG

# Import fallback implementations
from . import _fallback_impl


def matmul(x: torch.Tensor, w: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    T4-optimized matrix multiplication.
    
    Uses FP16 for T4 GPU tensor cores, falls back to FP32 for CPU.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        w: Weight tensor [d_model, d_out]
        *args: Additional arguments passed to the kernel
        **kwargs: Additional keyword arguments passed to the kernel
        
    Returns:
        Output tensor [batch_size, seq_len, d_out]
    """
    # Use FP16 for T4 GPU tensor cores
    if SYSTEM_CONFIG.has_tensor_cores and x.is_cuda:
        return _fallback_impl.matmul_fp16(x, w, *args, **kwargs)
    else:
        # Fallback to FP32 for CPU or non-tensor-core operations
        return _fallback_impl.matmul_fp32(x, w, *args, **kwargs)


def matmul_fp16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """FP16 matmul optimized for T4 GPU tensor cores."""
    return _fallback_impl.matmul_fp16(x, w)


def matmul_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """FP32 matmul for maximum precision."""
    return _fallback_impl.matmul_fp32(x, w)


def matmul_with_info(x: torch.Tensor, w: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, str]:
    """
    Matrix multiplication with kernel information.
    
    Returns both the result and the name of the kernel used.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        w: Weight tensor [d_model, d_out]
        *args: Additional arguments passed to the kernel
        **kwargs: Additional keyword arguments passed to the kernel
        
    Returns:
        Tuple of (output_tensor, kernel_name)
    """
    if SYSTEM_CONFIG.has_tensor_cores and x.is_cuda:
        result = _fallback_impl.matmul_fp16(x, w, *args, **kwargs)
        return result, "fp16_tensor_cores"
    else:
        result = _fallback_impl.matmul_fp32(x, w, *args, **kwargs)
        return result, "fp32_generic"


def print_kernel_info():
    """Print information about available kernels for debugging."""
    config = SYSTEM_CONFIG
    
    print("🔧 T4-Optimized Matmul Kernels:")
    print(f"   Architecture: {config.architecture}")
    print(f"   Compute Capability: {config.capability}")
    print(f"   Tensor Cores: {'✅' if config.has_tensor_cores else '❌'}")
    print(f"   Available kernels:")
    print(f"   ✅ fp16_tensor_cores: Optimized for T4 GPU")
    print(f"   📋 fp32_generic: Fallback for CPU/precision")


if __name__ == "__main__":
    print_kernel_info()
