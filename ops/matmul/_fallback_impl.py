"""
Fallback matmul implementations for generic hardware.

This module provides robust, hardware-agnostic matmul implementations
that work on any GPU or CPU, ensuring the system always has a working
fallback option.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def matmul_generic(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Generic matmul implementation that works on any hardware.
    
    This is the most robust fallback implementation that will work
    on any GPU or CPU, though it may not be optimally fast.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        w: Weight tensor [d_model, d_out]
        
    Returns:
        Output tensor [batch_size, seq_len, d_out]
    """
    # Use standard PyTorch matmul - works everywhere
    return torch.matmul(x, w.T)


def matmul_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    FP32 matmul for maximum numerical precision.
    
    Useful when numerical stability is more important than speed,
    or when running on older hardware without tensor cores.
    """
    # Ensure FP32 precision
    x_fp32 = x.to(torch.float32)
    w_fp32 = w.to(torch.float32)
    
    return torch.matmul(x_fp32, w_fp32.T)


def matmul_fp16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    FP16 matmul for better memory efficiency.
    
    Good compromise between speed and memory usage on older GPUs.
    """
    # Ensure FP16 precision
    x_fp16 = x.to(torch.float16)
    w_fp16 = w.to(torch.float16)
    
    return torch.matmul(x_fp16, w_fp16.T)


def matmul_cpu_optimized(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    CPU-optimized matmul implementation.
    
    Uses optimized CPU kernels when running on CPU.
    """
    if not x.is_cuda and not w.is_cuda:
        # Use optimized CPU implementation
        return torch.matmul(x, w.T)
    else:
        # Fall back to generic implementation
        return matmul_generic(x, w)


def matmul_with_memory_efficiency(x: torch.Tensor, w: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """
    Memory-efficient matmul using chunking.
    
    Processes the matmul in chunks to reduce memory usage,
    useful for very large tensors.
    """
    batch_size, seq_len, d_model = x.shape
    d_out = w.shape[0]
    
    # Initialize output tensor
    output = torch.zeros(batch_size, seq_len, d_out, device=x.device, dtype=x.dtype)
    
    # Process in chunks to save memory
    for i in range(0, d_out, chunk_size):
        end_idx = min(i + chunk_size, d_out)
        w_chunk = w[i:end_idx]
        
        # Compute chunk
        chunk_output = torch.matmul(x, w_chunk.T)
        output[:, :, i:end_idx] = chunk_output
    
    return output


def matmul_with_gradient_scaling(x: torch.Tensor, w: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Matmul with gradient scaling for numerical stability.
    
    Scales the result to prevent gradient overflow/underflow.
    """
    result = torch.matmul(x, w.T)
    return result * scale_factor


def get_safe_precision(x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.dtype, torch.dtype]:
    """
    Determine safe precision for the given tensors.
    
    Returns dtypes that won't cause overflow or underflow
    for the given tensor values.
    """
    # Check tensor ranges to determine safe precision
    x_max = x.abs().max().item()
    w_max = w.abs().max().item()
    
    # Choose precision based on tensor ranges
    if x_max > 65504 or w_max > 65504:  # FP16 max value
        return torch.float32, torch.float32
    elif x_max > 3.4e38 or w_max > 3.4e38:  # FP32 max value
        return torch.float64, torch.float64
    else:
        return torch.float16, torch.float16


def matmul_adaptive_precision(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Adaptive precision matmul that chooses the best dtype automatically.
    
    Analyzes the input tensors and chooses the optimal precision
    to balance speed and numerical stability.
    """
    input_dtype, weight_dtype = get_safe_precision(x, w)
    
    x_typed = x.to(input_dtype)
    w_typed = w.to(weight_dtype)
    
    return torch.matmul(x_typed, w_typed.T)
