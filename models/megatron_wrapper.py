"""
Megatron-LM wrapper for Blueberry LLM models.

This module provides a minimal wrapper that makes existing Blueberry models
compatible with Megatron-LM distributed training while preserving all
existing functionality.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from configs import AdaptiveMoEModelConfig


class MegatronWrapper(nn.Module):
    """
    Minimal wrapper that makes existing Blueberry models Megatron-compatible.
    
    This wrapper preserves the existing model architecture completely and only
    adds Megatron distributed training capabilities on top.
    """
    
    def __init__(self, blueberry_model: nn.Module, config: AdaptiveMoEModelConfig):
        """
        Initialize Megatron wrapper around existing Blueberry model.
        
        Args:
            blueberry_model: Existing Blueberry model (AdaptiveMoEMinimalLLM or AdaptiveStandardLLM)
            config: Model configuration with Megatron settings
        """
        super().__init__()
        self.model = blueberry_model  # Preserve existing model completely
        self.config = config
        
        # Initialize Megatron parallelism
        self._init_megatron()
        
        print(f"🚀 MegatronWrapper initialized with {type(blueberry_model).__name__}")
        print(f"   📊 Tensor parallel size: {self.config.tensor_parallel_size}")
    
    def _init_megatron(self):
        """Initialize Megatron distributed training."""
        try:
            # Import Megatron components
            from megatron.core import mpu
            from megatron.core.model_parallel_config import ModelParallelConfig
            
            # Check if distributed training is initialized
            if not torch.distributed.is_initialized():
                print("⚠️ Distributed training not initialized, skipping Megatron parallelism")
                return
            
            # Initialize model parallel groups if not already initialized
            if not mpu.model_parallel_is_initialized():
                mpu.initialize_model_parallel(
                    tensor_model_parallel_size=self.config.tensor_parallel_size,
                    pipeline_model_parallel_size=self.config.pipeline_parallel_size
                )
                print("✅ Megatron model parallelism initialized")
                print(f"   Tensor parallel size: {self.config.tensor_parallel_size}")
                print(f"   Pipeline parallel size: {self.config.pipeline_parallel_size}")
                print(f"   Model parallel rank: {mpu.get_tensor_model_parallel_rank()}")
                print(f"   Model parallel world size: {mpu.get_tensor_model_parallel_world_size()}")
            
            # Store the config for use in layer wrapping
            self.megatron_config = ModelParallelConfig()
            
            # Wrap model layers with Megatron parallel layers if needed
            self._wrap_parallel_layers()
            
        except ImportError as e:
            print(f"⚠️ Megatron initialization failed: {e}")
            print("   Falling back to standard distributed training")
    
    def _wrap_parallel_layers(self):
        """
        Wrap existing model layers with Megatron parallel layers.
        
        This preserves the existing model structure while adding parallelism.
        """
        try:
            from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
            from megatron.core import mpu
            
            # Wrap key layers for tensor parallelism
            self._wrap_linear_layers()
            
        except ImportError:
            # Megatron not available, use standard training
            pass
    
    def _wrap_linear_layers(self):
        """Wrap linear layers with Megatron parallel layers."""
        try:
            from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
            from megatron.core import mpu
            from megatron.core.model_parallel_config import ModelParallelConfig
            
            print("🔧 Wrapping layers with tensor parallelism...")
            
            # Use the stored Megatron config
            megatron_config = getattr(self, 'megatron_config', ModelParallelConfig())
            
            # Wrap the language modeling head for tensor parallelism
            if hasattr(self.model, 'lm_head'):
                original_lm_head = self.model.lm_head
                
                # Create parallel linear layer with proper config
                def identity_init(tensor):
                    return tensor
                
                parallel_lm_head = ColumnParallelLinear(
                    input_size=original_lm_head.in_features,
                    output_size=original_lm_head.out_features,
                    config=megatron_config,
                    init_method=identity_init,
                    bias=original_lm_head.bias is not None,
                    gather_output=True  # Gather output from all ranks
                )
                
                # Copy weights
                with torch.no_grad():
                    parallel_lm_head.weight.copy_(original_lm_head.weight)
                    if original_lm_head.bias is not None:
                        parallel_lm_head.bias.copy_(original_lm_head.bias)
                
                # Replace the layer
                self.model.lm_head = parallel_lm_head
                print("✅ Wrapped lm_head with tensor parallelism")
            
            # Wrap attention layers for better parallelism
            if hasattr(self.model, 'transformer_blocks'):
                for i, block in enumerate(self.model.transformer_blocks):
                    if hasattr(block, 'attention'):
                        # Wrap attention projection layers
                        if hasattr(block.attention, 'qkv_proj'):
                            original_qkv = block.attention.qkv_proj
                            parallel_qkv = ColumnParallelLinear(
                                input_size=original_qkv.in_features,
                                output_size=original_qkv.out_features,
                                config=megatron_config,
                                init_method=identity_init,
                                bias=original_qkv.bias is not None,
                                gather_output=False
                            )
                            with torch.no_grad():
                                parallel_qkv.weight.copy_(original_qkv.weight)
                                if original_qkv.bias is not None:
                                    parallel_qkv.bias.copy_(original_qkv.bias)
                            block.attention.qkv_proj = parallel_qkv
                            print(f"✅ Wrapped attention qkv_proj in block {i}")
                        
                        if hasattr(block.attention, 'out_proj'):
                            original_out = block.attention.out_proj
                            parallel_out = RowParallelLinear(
                                input_size=original_out.in_features,
                                output_size=original_out.out_features,
                                config=megatron_config,
                                init_method=identity_init,
                                bias=original_out.bias is not None,
                                input_is_parallel=True
                            )
                            with torch.no_grad():
                                parallel_out.weight.copy_(original_out.weight)
                                if original_out.bias is not None:
                                    parallel_out.bias.copy_(original_out.bias)
                            block.attention.out_proj = parallel_out
                            print(f"✅ Wrapped attention out_proj in block {i}")
            
            print("✅ Layer wrapping completed")
            
        except Exception as e:
            print(f"⚠️ Layer wrapping failed: {e}")
            print("   Continuing with standard distributed training")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the wrapped model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            return_aux_loss: Whether to return auxiliary losses
            
        Returns:
            Same as underlying model: logits and optional auxiliary loss
        """
        # Use existing model's forward pass unchanged
        return self.model(input_ids, return_aux_loss)
    
    def generate(self, *args, **kwargs):
        """Generation method - delegate to underlying model."""
        return self.model.generate(*args, **kwargs)
    
    def get_num_params(self, *args, **kwargs):
        """Get number of parameters - delegate to underlying model."""
        return self.model.get_num_params(*args, **kwargs)
    
    def estimate_mfu(self, *args, **kwargs):
        """Estimate MFU - delegate to underlying model."""
        return self.model.estimate_mfu(*args, **kwargs)
    
    def state_dict(self, *args, **kwargs):
        """State dict - delegate to underlying model."""
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        """Load state dict - delegate to underlying model."""
        return self.model.load_state_dict(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate any other attribute access to the underlying model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def create_megatron_model(config: AdaptiveMoEModelConfig, model_type: str = "moe") -> MegatronWrapper:
    """
    Create a Megatron-wrapped model.
    
    Args:
        config: Model configuration
        model_type: Type of model to create ("moe", "standard")
        
    Returns:
        MegatronWrapper instance
    """
    # Import the native model creation functions
    from .adaptive_llm import AdaptiveMoEMinimalLLM, AdaptiveStandardLLM
    
    # Create the underlying Blueberry model first
    if model_type == "moe":
        base_model = AdaptiveMoEMinimalLLM(config)
    elif model_type == "standard":
        base_model = AdaptiveStandardLLM(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Wrap it with Megatron capabilities
    megatron_model = MegatronWrapper(base_model, config)
    
    return megatron_model


def is_megatron_model(model: nn.Module) -> bool:
    """Check if a model is wrapped with Megatron capabilities."""
    return isinstance(model, MegatronWrapper)
