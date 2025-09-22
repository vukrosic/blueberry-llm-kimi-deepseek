"""
Configuration for T4-optimized Mixture of Experts model.

This module contains the configuration dataclass for the GPU-adaptive
MoE model with automatic optimization based on hardware capabilities.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from system import SYSTEM_CONFIG


@dataclass
class T4MoEModelConfig:
    """
    Configuration for the T4-optimized Mixture of Experts model.
    
    This configuration automatically adapts to the detected GPU architecture
    and enables optimizations like FP8 training on supported hardware.
    """
    
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 1000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # MoE specific parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01

    # GPU-adaptive parameters (optimized for T4)
    use_fp16_matmul: bool = True  # Use FP16 matmul operations for T4
    
    # Single GPU training parameters (T4 optimized)

    def __post_init__(self):
        """Post-initialization to validate and adapt configuration based on hardware."""
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Auto-detect optimal settings based on T4 GPU
        if SYSTEM_CONFIG.architecture == "t4":
            print("🚀 T4 GPU detected - using FP16 optimization")
        else:
            print("📋 Using FP16 precision (T4 optimized)")
        
        # Single T4 GPU - no distributed training needed
    
    def get_optimal_dtype(self):
        """Get the optimal data type for this configuration."""
        return SYSTEM_CONFIG.get_optimal_dtype()
    
    def supports_feature(self, feature: str) -> bool:
        """Check if the current configuration supports a specific feature."""
        feature_map = {
            "fp16_matmul": self.use_fp16_matmul,
            "tensor_cores": SYSTEM_CONFIG.has_tensor_cores,
        }
        return feature_map.get(feature, False)
    
    def get_info(self):
        """Get a dictionary with configuration information."""
        return {
            "model_type": "T4MoE",
            "architecture": f"{self.d_model}d-{self.n_layers}L-{self.n_heads}H",
            "moe": f"{self.num_experts}experts-top{self.expert_top_k}",
            "training": f"{self.max_steps}steps-bs{self.batch_size}",
            "gpu_features": {
                "fp16_matmul": self.supports_feature("fp16_matmul"),
                "tensor_cores": self.supports_feature("tensor_cores"),
            },
            "optimal_dtype": str(self.get_optimal_dtype()),
        }
    
    def print_config(self):
        """Print a formatted configuration summary."""
        info = self.get_info()
        print(f"\n📋 {info['model_type']} Configuration:")
        print(f"   Architecture: {info['architecture']}")
        print(f"   MoE: {info['moe']}")
        print(f"   Training: {info['training']}")
        print(f"   Data: {self.max_tokens:,} tokens, seq_len {self.max_seq_len}")
        print(f"   GPU Features:")
        for feature, enabled in info['gpu_features'].items():
            status = "✅" if enabled else "❌"
            print(f"     {status} {feature}")
        print(f"   Optimal Dtype: {info['optimal_dtype']}")


def get_t4_optimized_config() -> T4MoEModelConfig:
    """Get an optimized configuration for T4 GPU."""
    return T4MoEModelConfig(
        d_model=384,  # Balanced for T4 memory
        n_heads=8,    # Optimal for T4
        n_layers=6,  # Balanced for T4
        d_ff=1536,   # Balanced for T4
        batch_size=12,  # Optimized for T4 memory
        max_steps=2000,  # Good training length
        max_seq_len=1024,  # Good sequence length for T4
        num_experts=8,  # Good expert count for T4
        expert_top_k=2,
        gradient_accumulation_steps=3,  # Balanced for T4
        use_fp16_matmul=True,
        muon_lr=0.01,
        eval_every=250,
        eval_steps=50,
    )

