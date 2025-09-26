"""
Experiment 3: Advanced DeepSeek Configuration Import

Configuration management for Advanced DeepSeek attention features experiment.
Defines different DeepSeek attention configurations with advanced features.
"""

from typing import Dict, Any, List
from configs.moe_config import MoEModelConfig


def get_advanced_deepseek_configs() -> List[Dict[str, Any]]:
    """Get different Advanced DeepSeek configurations to test"""
    return [
        {
            'name': 'DeepSeek-Q-LoRA',
            'params': {
                'd_model': 512,
                'n_layers': 6,
                'd_ff': 2048,
                'num_experts': 8,
                'top_k': 2,
                'q_lora_rank': 64,
                'kv_lora_rank': 32,
                'attention_bias': True,
                'attn_implementation': 'eager'
            }
        },
        {
            'name': 'DeepSeek-Flash-Attention',
            'params': {
                'd_model': 512,
                'n_layers': 6,
                'd_ff': 2048,
                'num_experts': 8,
                'top_k': 2,
                'q_lora_rank': None,
                'kv_lora_rank': 64,
                'attention_bias': False,
                'attn_implementation': 'flash_attention_2'
            }
        },
        {
            'name': 'DeepSeek-Mixed-Heads',
            'params': {
                'd_model': 512,
                'n_layers': 6,
                'd_ff': 2048,
                'num_experts': 8,
                'top_k': 2,
                'q_lora_rank': 32,
                'kv_lora_rank': 48,
                'v_head_dim': 64,
                'qk_rope_head_dim': 32,
                'qk_nope_head_dim': 16,
                'attention_bias': True,
                'attn_implementation': 'eager'
            }
        },
        {
            'name': 'DeepSeek-Advanced-RoPE',
            'params': {
                'd_model': 512,
                'n_layers': 6,
                'd_ff': 2048,
                'num_experts': 8,
                'top_k': 2,
                'q_lora_rank': 48,
                'kv_lora_rank': 64,
                'rope_theta': 50000.0,
                'rope_scaling': {'type': 'linear', 'factor': 2.0},
                'attention_bias': False,
                'attn_implementation': 'eager'
            }
        },
        {
            'name': 'DeepSeek-Hybrid-LoRA',
            'params': {
                'd_model': 512,
                'n_layers': 6,
                'd_ff': 2048,
                'num_experts': 8,
                'top_k': 2,
                'q_lora_rank': 32,
                'kv_lora_rank': 32,
                'v_head_dim': 48,
                'qk_rope_head_dim': 40,
                'qk_nope_head_dim': 8,
                'attention_bias': True,
                'attn_implementation': 'flash_attention_2'
            }
        }
    ]


def get_training_configs() -> List[Dict[str, Any]]:
    """Get training configurations for each DeepSeek model"""
    return [
        {
            'batch_size': 4,
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
            'batch_size': 4,
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
            'batch_size': 4,
            'learning_rate': 8e-5,
            'weight_decay': 0.01,
            'num_epochs': 5
        }
    ]


def create_moe_config_from_deepseek(deepseek_config: Dict[str, Any]) -> MoEModelConfig:
    """Create MoE config from DeepSeek configuration"""
    params = deepseek_config['params']
    
    return MoEModelConfig(
        vocab_size=1000,
        d_model=params['d_model'],
        n_layers=params['n_layers'],
        d_ff=params['d_ff'],
        num_experts=params['num_experts'],
        top_k=params['top_k'],
        max_seq_len=512,
        dropout=0.1,
        # DeepSeek-specific parameters
        q_lora_rank=params.get('q_lora_rank'),
        kv_lora_rank=params.get('kv_lora_rank'),
        v_head_dim=params.get('v_head_dim'),
        qk_rope_head_dim=params.get('qk_rope_head_dim'),
        qk_nope_head_dim=params.get('qk_nope_head_dim'),
        attention_bias=params.get('attention_bias', False),
        attn_implementation=params.get('attn_implementation', 'eager'),
        rope_theta=params.get('rope_theta', 10000.0),
        rope_scaling=params.get('rope_scaling')
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
