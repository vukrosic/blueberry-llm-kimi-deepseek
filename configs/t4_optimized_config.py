"""
Optimized configuration for Tesla T4 GPU in Google Colab.

This configuration maximizes T4 resources while maintaining stability.
T4 has 16GB memory, Tensor Cores, and Turing architecture (compute capability 7.5).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from system import SYSTEM_CONFIG


@dataclass
class T4OptimizedConfig:
    """
    Optimized configuration for Tesla T4 GPU.
    
    This configuration is designed to maximize T4's 16GB memory and Tensor Core capabilities
    while maintaining training stability and convergence.
    """
    
    # Model architecture - larger than default to utilize more memory
    d_model: int = 512  # Increased from 256
    n_heads: int = 8    # Increased from 4
    n_layers: int = 8   # Increased from 4
    d_ff: int = 2048    # Increased from 1024
    batch_size: int = 16  # Increased from 8
    max_steps: int = 2000  # Increased from 1000

    # Training parameters
    gradient_accumulation_steps: int = 2  # Reduced from 4 to allow larger batch_size
    muon_lr: float = 0.01

    # Data parameters - longer sequences to better utilize memory
    max_seq_len: int = 1024  # Increased from 512
    num_documents: int = 2000  # Increased from 1000
    max_tokens: int = 200000  # Increased from 100000

    # Evaluation
    eval_every: int = 250
    eval_steps: int = 50

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (500, 1000, 1500, 2000)

    # MoE specific parameters - more experts for better capacity
    num_experts: int = 8  # Increased from 4
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01

    # GPU-adaptive parameters
    use_fp8: bool = False  # T4 doesn't support FP8
    use_adaptive_matmul: bool = True  # Use adaptive matmul operations
    
    # Megatron-LM distributed training parameters
    use_megatron: bool = False  # Single GPU setup
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    def __post_init__(self):
        """Post-initialization to validate and adapt configuration for T4."""
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # T4-specific optimizations
        if SYSTEM_CONFIG.architecture == "turing" and SYSTEM_CONFIG.memory_gb >= 14:
            print("🚀 T4 optimization enabled")
            print(f"   📈 Model: {self.d_model}d × {self.n_layers}L × {self.n_heads}H")
            print(f"   🧠 Experts: {self.num_experts}")
            print(f"   📊 Batch: {self.batch_size} (accum: {self.gradient_accumulation_steps})")
            print(f"   📝 Sequence: {self.max_seq_len}")
            print(f"   💾 Memory target: ~12-14GB (out of {SYSTEM_CONFIG.memory_gb:.1f}GB)")
        else:
            print("⚠️  T4 optimization not detected, using default settings")
    
    def get_optimal_dtype(self):
        """Get the optimal data type for T4 (FP16)."""
        return torch.float16
    
    def supports_feature(self, feature: str) -> bool:
        """Check if T4 supports a specific feature."""
        feature_map = {
            "fp8": False,  # T4 doesn't support FP8
            "adaptive_matmul": self.use_adaptive_matmul,
            "tensor_cores": True,  # T4 has Tensor Cores
            "bf16": False,  # T4 doesn't support BF16
        }
        return feature_map.get(feature, False)
    
    def get_info(self):
        """Get a dictionary with T4 configuration information."""
        return {
            "model_type": "T4OptimizedMoE",
            "architecture": f"{self.d_model}d-{self.n_layers}L-{self.n_heads}H",
            "moe": f"{self.num_experts}experts-top{self.expert_top_k}",
            "training": f"{self.max_steps}steps-bs{self.batch_size}",
            "gpu_features": {
                "fp8": self.supports_feature("fp8"),
                "adaptive_matmul": self.supports_feature("adaptive_matmul"),
                "tensor_cores": self.supports_feature("tensor_cores"),
            },
            "optimal_dtype": "torch.float16",
            "memory_target_gb": "12-14",
        }
    
    def print_config(self):
        """Print a formatted T4 configuration summary."""
        info = self.get_info()
        print(f"\n🚀 T4 Optimized Configuration:")
        print(f"   Architecture: {info['architecture']}")
        print(f"   MoE: {info['moe']}")
        print(f"   Training: {info['training']}")
        print(f"   Data: {self.max_tokens:,} tokens, seq_len {self.max_seq_len}")
        print(f"   Memory Target: {info['memory_target_gb']}GB")
        print(f"   GPU Features:")
        for feature, enabled in info['gpu_features'].items():
            status = "✅" if enabled else "❌"
            print(f"     {status} {feature}")
        print(f"   Optimal Dtype: {info['optimal_dtype']}")


def get_t4_optimized_config() -> T4OptimizedConfig:
    """Get the optimized T4 configuration."""
    return T4OptimizedConfig()


# Import torch for dtype reference
import torch
