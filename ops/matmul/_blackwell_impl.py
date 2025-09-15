"""
Blackwell-specific matmul implementations optimized for FP8.

This module contains highly optimized matmul kernels that leverage
Blackwell's native FP8 support and advanced tensor cores.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def matmul_fp8(x: torch.Tensor, w: torch.Tensor, x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0) -> torch.Tensor:
    """
    FP8 matmul optimized for Blackwell architecture.
    
    This implementation uses torch._scaled_mm for native FP8 acceleration
    on Blackwell GPUs with compute capability >= 9.0.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        w: Weight tensor [d_model, d_out]
        x_s: Input scaling factor
        w_s: Weight scaling factor  
        grad_s: Gradient scaling factor
        
    Returns:
        Output tensor [batch_size, seq_len, d_out]
    """
    # Ensure tensors are contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    
    # Reshape for matmul: [B*T, D] @ [D, D_out]
    original_shape = x.shape
    x_flat = x.view(-1, x.size(-1))
    
    # Convert to FP8 with scaling
    x_f8 = x_flat.div(x_s).to(torch.float8_e4m3fn)
    w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
    
    # Check if dimensions are compatible with torch._scaled_mm
    # torch._scaled_mm requires dimensions to be divisible by 16
    if x_f8.size(-1) % 16 == 0 and w_f8.size(0) % 16 == 0:
        # Use Blackwell's native FP8 matmul
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
    else:
        # Fallback to standard FP8 matmul for incompatible dimensions
        out = torch.matmul(x_f8.to(torch.bfloat16), w_f8.to(torch.bfloat16).T)
        out = out * (x_s * w_s)
    
    # Reshape back to original batch dimensions
    return out.reshape(*original_shape[:-1], -1)


def matmul_fp8_with_grad(x: torch.Tensor, w: torch.Tensor, x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FP8 matmul with gradient computation support.
    
    Returns the output tensor along with the FP8 versions of inputs
    needed for backward pass.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        w: Weight tensor [d_model, d_out]
        x_s: Input scaling factor
        w_s: Weight scaling factor
        grad_s: Gradient scaling factor
        
    Returns:
        Tuple of (output, x_f8, w_f8) where output is the result and
        x_f8, w_f8 are the FP8 versions needed for gradients
    """
    # Ensure tensors are contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    
    # Reshape for matmul
    original_shape = x.shape
    x_flat = x.view(-1, x.size(-1))
    
    # Convert to FP8
    x_f8 = x_flat.div(x_s).to(torch.float8_e4m3fn)
    w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
    
    # FP8 matmul
    out = torch._scaled_mm(
        x_f8,
        w_f8.T,
        out_dtype=torch.bfloat16,
        scale_a=x.new_tensor(x_s, dtype=torch.float32),
        scale_b=x.new_tensor(w_s, dtype=torch.float32),
        use_fast_accum=True,
    )
    
    # Reshape output
    output = out.reshape(*original_shape[:-1], -1)
    
    return output, x_f8, w_f8


def matmul_bf16_tensor_cores(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    BF16 matmul using Blackwell's advanced tensor cores.
    
    Fallback implementation when FP8 is not available or desired.
    Uses BF16 precision with tensor core acceleration.
    """
    # Ensure BF16 precision
    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = w.to(torch.bfloat16)
    
    # Standard matmul - PyTorch will automatically use tensor cores
    return torch.matmul(x_bf16, w_bf16.T)


def get_optimal_scaling_factors(x: torch.Tensor, w: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute optimal scaling factors for FP8 quantization.
    
    This helps maintain numerical stability by ensuring the input
    and weight tensors are properly scaled before FP8 conversion.
    """
    # Compute scaling factors to avoid overflow
    x_max = x.abs().max().item()
    w_max = w.abs().max().item()
    
    # FP8 E4M3 has range of ~[-448, 448]
    fp8_max = 448.0
    
    x_s = fp8_max / max(x_max, 1e-8)
    w_s = fp8_max / max(w_max, 1e-8)
    
    # Gradient scaling factor
    grad_s = 1.0 / (x_s * w_s)
    
    return x_s, w_s, grad_s
