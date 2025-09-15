"""
Muon optimizer implementation.

Muon (MomentUm Orthogonalized by Newton-schulz) is an optimizer designed
for training large language models with improved convergence properties.

Reference: https://kellerjordan.github.io/posts/muon/
"""

import torch
from typing import Iterator


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    
    This function computes an approximation to G/||G||_F using Newton-Schulz iteration.
    The quintic iteration is designed to maximize the slope at zero for faster convergence.
    
    Args:
        G: Input tensor to orthogonalize (must be at least 2D)
        steps: Number of Newton-Schulz iteration steps
        
    Returns:
        Orthogonalized tensor of the same shape as G
    """
    assert G.ndim >= 2, "Input tensor must be at least 2D"
    
    # Coefficients for quintic Newton-Schulz iteration
    # These are selected to maximize the slope at zero
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Work in bfloat16 for stability and speed
    X = G.bfloat16()

    # Handle tall matrices by transposing to wide form
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize to ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # Quintic computation
        X = a * X + B @ X

    # Transpose back if we transposed initially
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    This optimizer combines SGD-momentum with orthogonalization via Newton-Schulz
    iteration. It's particularly effective for training large language models.
    
    Key features:
    - Uses Newton-Schulz iteration for orthogonalization
    - Stable in bfloat16 precision
    - Adaptive learning rate based on parameter shape
    - Nesterov momentum support
    
    Warning: This optimizer should only be used for 2D weight matrices.
    Use AdamW for embeddings, biases, and normalization parameters.
    """
    
    def __init__(
        self, 
        params: Iterator[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0
    ):
        """
        Initialize Muon optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            momentum: Momentum coefficient
            nesterov: Whether to use Nesterov momentum
            ns_steps: Number of Newton-Schulz iteration steps
            weight_decay: Weight decay coefficient
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov, 
            ns_steps=ns_steps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                
                # Ensure parameter is 2D (Muon is designed for weight matrices)
                if grad.ndim != 2:
                    raise ValueError(f"Muon optimizer requires 2D parameters, got {grad.ndim}D")

                state = self.state[p]

                # Initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                
                # Apply weight decay to parameter
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                
                # Update momentum buffer
                buf.lerp_(grad, 1 - group["momentum"])
                
                # Choose gradient for update (Nesterov vs standard momentum)
                if group["nesterov"]:
                    update_grad = grad.lerp_(buf, group["momentum"])
                else:
                    update_grad = buf
                
                # Orthogonalize the gradient using Newton-Schulz iteration
                orthogonal_grad = zeropower_via_newtonschulz5(update_grad, steps=group["ns_steps"])
                
                # Adaptive learning rate based on parameter shape
                # This helps balance updates for different sized matrices
                adaptive_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5
                
                # Apply update
                p.add_(orthogonal_grad.view_as(p), alpha=-adaptive_lr)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """
        Zero gradients of all parameters.
        
        Args:
            set_to_none: If True, set gradients to None instead of zero
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_().zero_()

    def get_lr_multiplier(self, param: torch.Tensor) -> float:
        """
        Get the learning rate multiplier for a parameter based on its shape.
        
        Args:
            param: Parameter tensor
            
        Returns:
            Learning rate multiplier
        """
        if param.ndim == 2:
            return max(1, param.size(-2) / param.size(-1)) ** 0.5
        else:
            return 1.0

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        return f"Muon(lr={self.defaults['lr']}, momentum={self.defaults['momentum']}, nesterov={self.defaults['nesterov']})"


class MuonWithWarmup(Muon):
    """
    Muon optimizer with momentum warmup.
    
    This variant gradually increases momentum from a lower value to the target
    momentum over a specified number of steps. This can improve training stability.
    """
    
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        momentum_warmup_steps: int = 1000,
        initial_momentum: float = 0.85,
        **kwargs
    ):
        """
        Initialize Muon with momentum warmup.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            momentum: Target momentum coefficient
            momentum_warmup_steps: Number of steps to warm up momentum
            initial_momentum: Initial momentum value
            **kwargs: Additional arguments passed to Muon
        """
        super().__init__(params, lr=lr, momentum=momentum, **kwargs)
        self.momentum_warmup_steps = momentum_warmup_steps
        self.initial_momentum = initial_momentum
        self.target_momentum = momentum
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform optimization step with momentum warmup.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        # Update momentum based on warmup schedule
        if self.step_count < self.momentum_warmup_steps:
            frac = self.step_count / self.momentum_warmup_steps
            current_momentum = (1 - frac) * self.initial_momentum + frac * self.target_momentum
            
            # Update momentum in all parameter groups
            for group in self.param_groups:
                group["momentum"] = current_momentum
        else:
            # Ensure we're at target momentum after warmup
            for group in self.param_groups:
                group["momentum"] = self.target_momentum
        
        self.step_count += 1
        
        # Call parent step method
        return super().step(closure)

    def get_current_momentum(self) -> float:
        """Get the current momentum value."""
        if self.step_count < self.momentum_warmup_steps:
            frac = self.step_count / self.momentum_warmup_steps
            return (1 - frac) * self.initial_momentum + frac * self.target_momentum
        else:
            return self.target_momentum
