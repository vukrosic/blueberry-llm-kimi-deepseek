#!/usr/bin/env python3
"""
Auto-configuration for Blueberry LLM
Detects hardware and automatically configures optimal training setup
"""

import os
import sys
import torch
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

@dataclass
class AutoConfig:
    """Auto-detected configuration"""
    # Hardware
    num_gpus: int
    gpu_memory_gb: float
    
    # Model (auto-scaled)
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    num_experts: int
    
    # Training (auto-optimized)
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    max_seq_len: int
    
    # Performance
    use_distributed: bool
    use_amp: bool
    use_megatron: bool

class BlueberryAutoConfigurator:
    """One class that does everything"""
    
    def __init__(self):
        self.config = self._detect_and_configure()
    
    def _detect_and_configure(self) -> AutoConfig:
        """Main auto-configuration logic"""
        
        # Detect hardware
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if num_gpus == 0:
            return self._cpu_config()
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        total_memory = gpu_memory_gb * num_gpus
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        print(f"🔍 Debug: GPU name = '{gpu_name}'")
        print(f"🔍 Debug: Total memory = {total_memory:.1f} GB")
        
        # RTX 4090 optimized configuration (primary configuration)
        if 'rtx 4090' in gpu_name or 'geforce rtx 4090' in gpu_name:
            print("🚀 RTX 4090 detected! Using optimized configuration")
            config = {
                'd_model': 768, 'n_layers': 16, 'n_heads': 16, 'd_ff': 3072,
                'num_experts': 16, 'batch_size': 24, 'max_seq_len': 2048
            }
        else:
            # Default configuration for other GPUs
            print("📋 Using default GPU configuration")
            config = {
                'd_model': 384, 'n_layers': 6, 'n_heads': 8, 'd_ff': 1536,
                'num_experts': 8, 'batch_size': 16, 'max_seq_len': 1024
            }
        
        # Set training parameters
        if 'rtx 4090' in gpu_name or 'geforce rtx 4090' in gpu_name:
            # RTX 4090 optimized training parameters
            gradient_accumulation_steps = 2  # Reduced for larger batch size
            max_steps = 2000  # More training steps
        else:
            gradient_accumulation_steps = max(1, 32 // config['batch_size'])
            max_steps = 1000
        
        return AutoConfig(
            num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb,
            **config,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            learning_rate=0.01,
            use_distributed=(num_gpus > 1),
            use_amp=True,
            use_megatron=False  # Default to False, can be overridden by flags
        )
    
    def _cpu_config(self) -> AutoConfig:
        """Minimal config for CPU-only systems"""
        return AutoConfig(
            num_gpus=0, gpu_memory_gb=0,
            d_model=128, n_layers=2, n_heads=4, d_ff=512, num_experts=2,
            batch_size=4, gradient_accumulation_steps=8, max_steps=1000,
            learning_rate=0.001, max_seq_len=256,
            use_distributed=False, use_amp=False, use_megatron=False
        )
    
    def print_config(self):
        """Print detected configuration"""
        print("🫐 Blueberry LLM Auto-Configuration")
        print("=" * 50)
        
        if self.config.num_gpus == 0:
            print("🖥️  Mode: CPU Training (Limited)")
        else:
            print(f"🚀 Mode: GPU Training ({self.config.num_gpus} GPUs)")
            print(f"   Memory: {self.config.gpu_memory_gb:.1f} GB per GPU")
        
        print(f"📏 Model: {self.config.d_model}d × {self.config.n_layers}L × {self.config.n_heads}H")
        print(f"🧠 Experts: {self.config.num_experts}")
        print(f"📊 Batch: {self.config.batch_size} (accum: {self.config.gradient_accumulation_steps})")
        print(f"📝 Sequence: {self.config.max_seq_len}")
        print(f"⚡ Mixed Precision: {'Yes' if self.config.use_amp else 'No'}")
        
        if self.config.use_distributed:
            print(f"🌐 Data Parallel: Yes (across {self.config.num_gpus} GPUs)")
            print(f"   Run with: torchrun --nproc_per_node={self.config.num_gpus} train_auto.py")
        
        print("=" * 50)
    
    def get_model_config(self):
        """Convert to MoEModelConfig format"""
        from legacy.llm import MoEModelConfig
        
        return MoEModelConfig(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            batch_size=self.config.batch_size,
            max_steps=self.config.max_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            muon_lr=self.config.learning_rate,
            max_seq_len=self.config.max_seq_len,
            num_experts=self.config.num_experts,
            use_amp=self.config.use_amp,
        )

def auto_configure() -> BlueberryAutoConfigurator:
    """One function call to auto-configure everything"""
    return BlueberryAutoConfigurator()

if __name__ == "__main__":
    # Demo the auto-configuration
    configurator = auto_configure()
    configurator.print_config()
