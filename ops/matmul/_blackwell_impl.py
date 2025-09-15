"""
Blackwell-specific matmul implementations optimized for FP8.

This module contains highly optimized matmul kernels that leverage
Blackwell's native FP8 support and advanced tensor cores.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


# Custom operators with proper gradient support for FP8 matmul
@torch.library.custom_op("blueberry::mm", mutates_args=())
def mm_op(x: torch.Tensor, w: torch.Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8 matmul with gradient support."""
    @torch.compile
    def impl(x: torch.Tensor, w: torch.Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)


@mm_op.register_fake
def _(x: torch.Tensor, w: torch.Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)


@torch.library.custom_op("blueberry::mm_backward", mutates_args=())
def mm_backward_op(g: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for FP8 matmul."""
    @torch.compile
    def impl(grad: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)


@mm_backward_op.register_fake
def _(g: torch.Tensor, x_f8: torch.Tensor, w_f8: torch.Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)


def backward(ctx, grad_out: torch.Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.blueberry.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None


def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)


mm_op.register_autograd(backward, setup_context=setup_context)


def matmul_fp8(x: torch.Tensor, w: torch.Tensor, x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0) -> torch.Tensor:
    """
    FP8 matmul optimized for Blackwell architecture with proper gradient support.
    
    This implementation uses custom operators with torch._scaled_mm for native FP8 acceleration
    on Blackwell GPUs with compute capability >= 9.0, with proper backward pass handling.
    
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
    
    # Check if dimensions are compatible with torch._scaled_mm
    # torch._scaled_mm requires dimensions to be divisible by 16
    if x_flat.size(-1) % 16 == 0 and w.size(0) % 16 == 0:
        # Use custom operator with proper gradient support
        if x.requires_grad or w.requires_grad:
            out = torch.ops.blueberry.mm(x_flat, w, x_s, w_s, grad_s)[0]
        else:
            # For inference, use the direct implementation
            x_f8 = x_flat.div(x_s).to(torch.float8_e4m3fn)
            w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
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
        x_f8 = x_flat.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch.matmul(x_f8.to(torch.bfloat16), w_f8.to(torch.bfloat16).T)
        out = out * (x_s * w_s)
    
    # Reshape back to original batch dimensions
    return out.reshape(*original_shape[:-1], -1)


def matmul_fp8_with_grad(x: torch.Tensor, w: torch.Tensor, x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FP8 matmul with gradient computation support using custom operators.
    
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
    
    # Use custom operator for gradient support
    out, x_f8, w_f8 = torch.ops.blueberry.mm(x_flat, w, x_s, w_s, grad_s)
    
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
