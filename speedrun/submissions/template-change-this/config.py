"""
T4 Speedrun Configuration

This module defines the constraints and configuration for the T4 speedrun challenge.
Participants can modify anything in the speedrun directory, but must adhere to these constraints.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from configs import AdaptiveMoEModelConfig


@dataclass
class T4SpeedrunConfig(AdaptiveMoEModelConfig):
    """
    T4 Speedrun Configuration with fixed constraints.
    
    This configuration enforces the speedrun rules while allowing participants
    to optimize within these constraints.
    """
    
    # SPEEDRUN CONSTRAINTS - DO NOT MODIFY
    SPEEDRUN_TIME_LIMIT_MINUTES: int = 5
    SPEEDRUN_DATASET_SEED: int = 42  # Fixed seed for reproducible dataset
    SPEEDRUN_VAL_SPLIT: float = 0.1  # Fixed validation split
    SPEEDRUN_MAX_TOKENS: int = 100000  # Fixed dataset size for fair comparison
    SPEEDRUN_MAX_DOCUMENTS: int = 1000  # Fixed number of documents
    
    # T4 Hardware Constraints (informational only - no enforcement)
    T4_MEMORY_GB: float = 16.0
    
    # Default configuration optimized for T4
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    batch_size: int = 16
    max_steps: int = 2000  # Participants can adjust this within time limit
    max_seq_len: int = 512
    num_documents: int = 1000
    max_tokens: int = 100000
    
    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01
    eval_every: int = 200
    eval_steps: int = 50
    
    # MoE parameters
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01
    
    # T4-specific optimizations
    use_amp: bool = True  # Essential for T4
    use_fp8: bool = False  # T4 doesn't support FP8
    use_adaptive_matmul: bool = True
    
    def __post_init__(self):
        """Initialize speedrun configuration."""
        # Basic validation
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Enforce only dataset constraints (for fair comparison)
        assert self.num_documents == self.SPEEDRUN_MAX_DOCUMENTS, f"Must use exactly {self.SPEEDRUN_MAX_DOCUMENTS} documents"
        assert self.max_tokens == self.SPEEDRUN_MAX_TOKENS, f"Must use exactly {self.SPEEDRUN_MAX_TOKENS} tokens"
        
        # Ensure T4 compatibility
        self.use_fp8 = False  # T4 doesn't support FP8
        print("🚀 T4 Speedrun Configuration Loaded")
        print(f"   ⏱️ Time Limit: {self.SPEEDRUN_TIME_LIMIT_MINUTES} minutes")
        print(f"   📊 Dataset: {self.max_tokens:,} tokens, {self.num_documents} documents")
        print(f"   🎯 Target: Lowest validation loss in {self.SPEEDRUN_TIME_LIMIT_MINUTES} minutes")
    
    def validate_speedrun_constraints(self) -> bool:
        """Validate that speedrun constraints are met."""
        constraints = [
            (self.num_documents == self.SPEEDRUN_MAX_DOCUMENTS, f"Documents {self.num_documents} != required {self.SPEEDRUN_MAX_DOCUMENTS}"),
            (self.max_tokens == self.SPEEDRUN_MAX_TOKENS, f"Tokens {self.max_tokens} != required {self.SPEEDRUN_MAX_TOKENS}"),
            (self.use_fp8 == False, "FP8 not supported on T4"),
            (self.d_model % self.n_heads == 0, f"d_model {self.d_model} not divisible by n_heads {self.n_heads}"),
        ]
        
        for constraint, message in constraints:
            if not constraint:
                print(f"❌ Speedrun constraint violated: {message}")
                return False
        
        print("✅ All speedrun constraints satisfied")
        return True
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in GB."""
        # Rough estimation based on model size and batch size
        model_params = (
            self.d_model * self.d_model * 4 * self.n_layers +  # Attention weights
            self.d_model * self.d_ff * 2 * self.n_layers +      # FFN weights
            self.num_experts * self.d_model * self.d_ff * 2 +   # MoE expert weights
            self.d_model * self.max_tokens // 1000              # Embedding weights
        )
        
        # Convert to GB (assuming float32)
        model_memory_gb = model_params * 4 / (1024**3)
        
        # Add batch memory (activations, gradients, optimizer states)
        batch_memory_gb = (
            self.batch_size * self.max_seq_len * self.d_model * 4 * 3 / (1024**3)  # Activations
        )
        
        total_memory_gb = model_memory_gb + batch_memory_gb
        
        print(f"📊 Estimated Memory Usage:")
        print(f"   Model: {model_memory_gb:.2f} GB")
        print(f"   Batch: {batch_memory_gb:.2f} GB")
        print(f"   Total: {total_memory_gb:.2f} GB / {self.T4_MEMORY_GB} GB ({total_memory_gb/self.T4_MEMORY_GB:.1%})")
        
        return total_memory_gb
    
    def get_speedrun_info(self):
        """Get speedrun-specific information."""
        return {
            "time_limit_minutes": self.SPEEDRUN_TIME_LIMIT_MINUTES,
            "dataset_size": f"{self.max_tokens:,} tokens, {self.num_documents} documents",
            "hardware": "Tesla T4 (16GB VRAM)",
            "constraints": {
                "max_batch_size": self.T4_MAX_BATCH_SIZE,
                "max_seq_len": self.T4_MAX_SEQ_LEN,
                "fp8_support": False,
                "fixed_dataset": True,
            },
            "optimization_target": "Lowest validation loss",
        }


def get_t4_speedrun_config() -> T4SpeedrunConfig:
    """Get the default T4 speedrun configuration."""
    return T4SpeedrunConfig()


def create_custom_t4_config(**kwargs) -> T4SpeedrunConfig:
    """
    Create a custom T4 speedrun configuration.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        Custom T4SpeedrunConfig instance
        
    Example:
        config = create_custom_t4_config(
            d_model=512,
            n_layers=8,
            batch_size=32,
            muon_lr=0.005
        )
    """
    config = T4SpeedrunConfig()
    
    # Apply custom parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"⚠️ Unknown parameter: {key}")
    
    # Re-run post_init to validate constraints
    config.__post_init__()
    
    return config


# Example configurations for different strategies
def get_memory_optimized_config() -> T4SpeedrunConfig:
    """Configuration optimized for memory efficiency."""
    return T4SpeedrunConfig(
        d_model=192,
        n_heads=6,
        n_layers=4,
        d_ff=768,
        batch_size=8,
        gradient_accumulation_steps=8,
        max_seq_len=256,
        num_experts=4,
    )


def get_performance_optimized_config() -> T4SpeedrunConfig:
    """Configuration optimized for maximum performance within constraints."""
    return T4SpeedrunConfig(
        d_model=384,
        n_heads=8,
        n_layers=8,
        d_ff=1536,
        batch_size=24,
        gradient_accumulation_steps=2,
        max_seq_len=512,
        num_experts=12,
        muon_lr=0.015,
    )


def get_balanced_config() -> T4SpeedrunConfig:
    """Balanced configuration between memory and performance."""
    return T4SpeedrunConfig(
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        batch_size=16,
        gradient_accumulation_steps=4,
        max_seq_len=512,
        num_experts=8,
        muon_lr=0.01,
    )
