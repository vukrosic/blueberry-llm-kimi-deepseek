#!/usr/bin/env python3
"""
T4-specific configuration for Blueberry LLM
Optimized for single Tesla T4 GPU training setup
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
    """T4-optimized configuration"""
    # Model (T4-optimized)
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    num_experts: int
    
    # Training (T4-optimized)
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    max_seq_len: int
    
    # Performance (T4-optimized)
    use_amp: bool

class BlueberryT4Configurator:
    """One class that does everything"""
    
    def __init__(self):
        self.config = self._detect_and_configure()
    
    def _detect_and_configure(self) -> AutoConfig:
        """Main auto-configuration logic for single T4 GPU"""
        
        # Detect hardware - expect single T4 GPU
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if num_gpus == 0:
            print("❌ No CUDA GPU detected. This version requires a T4 GPU.")
            raise RuntimeError("T4 GPU required but not found")
        
        if num_gpus > 1:
            print(f"⚠️ Multiple GPUs detected ({num_gpus}), but this version is optimized for single T4 GPU only")
            print("   Using only the first GPU")
        
        # Check for T4 GPU
        device_name = torch.cuda.get_device_name(0).lower()
        if "tesla t4" in device_name or "t4" in device_name:
            print("🚀 Tesla T4 detected - using optimized configuration")
        else:
            print(f"⚠️ Non-T4 GPU detected: {device_name}")
            print("   This version is optimized for T4 GPUs, but will attempt to run")
        
        return self._t4_optimized_config()  # Always use T4 config
    
    def _t4_optimized_config(self) -> AutoConfig:
        """Optimized config for single Tesla T4 GPU - balanced for memory efficiency"""
        return AutoConfig(
            d_model=384,  # Balanced for T4 memory
            n_layers=6,   # Balanced for T4
            n_heads=8,    # Optimal for T4
            d_ff=1536,    # Balanced for T4
            num_experts=8,  # Good for T4
            batch_size=12,  # Optimized for T4 memory
            gradient_accumulation_steps=3,  # Balanced for T4
            max_steps=2000,  # Good training length
            learning_rate=0.01,
            max_seq_len=1024,  # Good for T4
            use_amp=True  # FP16 for T4 tensor cores
        )
    
    def _cpu_config(self) -> AutoConfig:
        """Minimal config for CPU-only systems"""
        return AutoConfig(
            d_model=128, n_layers=2, n_heads=4, d_ff=512, num_experts=2,
            batch_size=4, gradient_accumulation_steps=8, max_steps=1000,
            learning_rate=0.001, max_seq_len=256,
            use_amp=False
        )
    
    def print_config(self):
        """Print detected configuration"""
        print("🫐 Blueberry LLM T4-Configuration")
        print("=" * 50)
        
        print("🚀 Mode: T4 GPU Training")
        
        print(f"📏 Model: {self.config.d_model}d × {self.config.n_layers}L × {self.config.n_heads}H")
        print(f"🧠 Experts: {self.config.num_experts}")
        print(f"📊 Batch: {self.config.batch_size} (accum: {self.config.gradient_accumulation_steps})")
        print(f"📝 Sequence: {self.config.max_seq_len}")
        print(f"⚡ Mixed Precision: {'Yes' if self.config.use_amp else 'No'}")
        
        print(f"🌐 Single T4 GPU Training")
        
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

def t4_configure() -> BlueberryT4Configurator:
    """One function call to configure T4-optimized training"""
    return BlueberryT4Configurator()

if __name__ == "__main__":
    # Demo the T4-configuration
    configurator = t4_configure()
    configurator.print_config()
