"""
System configuration and GPU detection module.

This module provides a singleton SystemConfig class that detects and caches
GPU capabilities at startup, enabling efficient runtime dispatch to optimized kernels.
"""

import torch
from typing import Tuple, Dict, Any


class SystemConfig:
    """
    Singleton class that detects and caches GPU system capabilities.
    
    This class is initialized once at startup and provides information about:
    - Number of available GPUs
    - GPU compute capability (major, minor)
    - Support for specific features (FP8, Tensor Cores, etc.)
    - Memory information
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._detect_gpu_capabilities()
            SystemConfig._initialized = True
    
    def _detect_gpu_capabilities(self):
        """Detect and cache GPU capabilities."""
        self.device_count = torch.cuda.device_count()
        
        if self.device_count > 0:
            # Get compute capability of the first GPU (assuming homogeneous cluster)
            self.capability = torch.cuda.get_device_capability(0)
            self.device_name = torch.cuda.get_device_name(0)
            
            # Feature detection based on compute capability (T4 optimized)
            self.has_tensor_cores = self.capability >= (7, 0)  # T4 has tensor cores
            
            # Memory information
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            self.memory_gb = self.total_memory / (1024**3)
            
            # Architecture classification
            self.architecture = self._classify_architecture()
            
        else:
            # No CUDA available
            self.capability = (0, 0)
            self.device_name = "CPU"
            self.has_tensor_cores = False
            self.total_memory = 0
            self.memory_gb = 0
            self.architecture = "cpu"
    
    def _classify_architecture(self) -> str:
        """Classify GPU architecture - optimized for T4 GPU."""
        major, minor = self.capability
        
        # T4 has compute capability 7.5
        if major == 7 and minor == 5:
            return "t4"
        else:
            # For non-T4 GPUs, return generic classification
            return "other"
    
    def get_optimal_dtype(self) -> torch.dtype:
        """Get the optimal data type for T4 GPU architecture."""
        return torch.float16  # T4 is optimized for FP16 with tensor cores
    
    def supports_feature(self, feature: str) -> bool:
        """Check if the current GPU supports a specific feature."""
        feature_map = {
            "tensor_cores": self.has_tensor_cores,
        }
        return feature_map.get(feature, False)
    
    def get_info(self) -> Dict[str, Any]:
        """Get a dictionary with all system information."""
        return {
            "device_count": self.device_count,
            "capability": self.capability,
            "device_name": self.device_name,
            "architecture": self.architecture,
            "memory_gb": self.memory_gb,
            "has_tensor_cores": self.has_tensor_cores,
            "optimal_dtype": str(self.get_optimal_dtype()),
        }
    
    def __repr__(self) -> str:
        return f"SystemConfig(arch={self.architecture}, capability={self.capability}, gpus={self.device_count})"


# Create a single global instance
SYSTEM_CONFIG = SystemConfig()


def get_system_config() -> SystemConfig:
    """Get the global system configuration instance."""
    return SYSTEM_CONFIG


def print_system_info():
    """Print detailed system information for debugging."""
    config = get_system_config()
    info = config.get_info()
    
    print("🔍 GPU System Configuration:")
    print(f"   Architecture: {info['architecture']}")
    print(f"   Device: {info['device_name']}")
    print(f"   Compute Capability: {info['capability']}")
    print(f"   GPU Count: {info['device_count']}")
    print(f"   Memory: {info['memory_gb']:.1f} GB")
    print(f"   Tensor Cores: {info['has_tensor_cores']}")
    print(f"   Optimal Dtype: {info['optimal_dtype']}")


if __name__ == "__main__":
    print_system_info()
