"""
Experiment 3: Mamba Configuration Import

Configuration management for Mamba State Space Model experiment.
Defines different Mamba configurations and training parameters.
"""

from typing import Dict, Any, List
from configs.moe_config import MoEModelConfig


def get_mamba_configs() -> List[Dict[str, Any]]:
    """Get different Mamba configurations to test"""
    return [
        {
            'name': 'Mamba-Small',
            'params': {
                'd_model': 256,
                'n_layers': 4,
                'd_ff': 1024,
                'num_experts': 4,
                'top_k': 2,
                'd_state': 16,
                'd_conv': 4,
                'expand': 2
            }
        },
        {
            'name': 'Mamba-Medium',
            'params': {
                'd_model': 512,
                'n_layers': 6,
                'd_ff': 2048,
                'num_experts': 8,
                'top_k': 2,
                'd_state': 16,
                'd_conv': 4,
                'expand': 2
            }
        },
        {
            'name': 'Mamba-Large',
            'params': {
                'd_model': 768,
                'n_layers': 8,
                'd_ff': 3072,
                'num_experts': 12,
                'top_k': 2,
                'd_state': 16,
                'd_conv': 4,
                'expand': 2
            }
        },
        {
            'name': 'Mamba-Wide',
            'params': {
                'd_model': 512,
                'n_layers': 4,
                'd_ff': 4096,
                'num_experts': 16,
                'top_k': 2,
                'd_state': 32,
                'd_conv': 8,
                'expand': 4
            }
        },
        {
            'name': 'Mamba-Deep',
            'params': {
                'd_model': 384,
                'n_layers': 12,
                'd_ff': 1536,
                'num_experts': 8,
                'top_k': 2,
                'd_state': 16,
                'd_conv': 4,
                'expand': 2
            }
        }
    ]


def get_training_configs() -> List[Dict[str, Any]]:
    """Get training configurations for each Mamba model"""
    return [
        {
            'batch_size': 8,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'num_epochs': 5
        },
        {
            'batch_size': 4,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'num_epochs': 5
        },
        {
            'batch_size': 2,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'num_epochs': 5
        },
        {
            'batch_size': 4,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'num_epochs': 5
        },
        {
            'batch_size': 6,
            'learning_rate': 8e-5,
            'weight_decay': 0.01,
            'num_epochs': 5
        }
    ]


def create_moe_config_from_mamba(mamba_config: Dict[str, Any]) -> MoEModelConfig:
    """Create MoE config from Mamba configuration"""
    params = mamba_config['params']
    
    return MoEModelConfig(
        vocab_size=1000,
        d_model=params['d_model'],
        n_layers=params['n_layers'],
        d_ff=params['d_ff'],
        num_experts=params['num_experts'],
        top_k=params['top_k'],
        max_seq_len=512,
        dropout=0.1,
        # Mamba-specific parameters
        d_state=params.get('d_state', 16),
        d_conv=params.get('d_conv', 4),
        expand=params.get('expand', 2)
    )


def create_config_from_moe_config(moe_config: MoEModelConfig) -> Dict[str, Any]:
    """Create training config from MoE config"""
    return {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 5,
        'warmup_steps': 100,
        'max_grad_norm': 1.0,
        'gradient_accumulation_steps': 1
    }
