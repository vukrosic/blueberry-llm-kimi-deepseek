#!/usr/bin/env python3
"""
Parameter count analysis for Experiment 8 models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from experiments.exp8.exp8_reduced_ablation_models import (
    create_reduced_ablation_model,
    REDUCED_ABLATION_MODELS
)

def analyze_parameter_counts():
    """Analyze parameter counts for all models"""
    print("🔍 Parameter Count Analysis for Experiment 8")
    print("=" * 60)
    
    # Create a config for testing
    config = MoEModelConfig(
        d_model=512,  # Will be overridden by individual models
        n_heads=8,
        n_layers=3,
        d_ff=2048,
        num_experts=8,
        expert_top_k=2,
        vocab_size=32000,
        max_seq_len=256,
        batch_size=16,
        max_steps=10,
        max_tokens=10000,
        num_documents=100,
    )
    
    results = {}
    
    for model_name in REDUCED_ABLATION_MODELS.keys():
        try:
            # Create a copy of config to avoid modifying the original
            model_config = MoEModelConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                d_ff=config.d_ff,
                num_experts=config.num_experts,
                expert_top_k=config.expert_top_k,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
                batch_size=config.batch_size,
                max_steps=config.max_steps,
                max_tokens=config.max_tokens,
                num_documents=config.num_documents,
            )
            
            model = create_reduced_ablation_model(model_name, model_config)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results[model_name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'params_millions': total_params / 1e6,
                'd_model': model_config.d_model,
                'num_experts': model_config.num_experts,
                'n_layers': model_config.n_layers
            }
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Print results
    print(f"{'Model':<35} {'Params (M)':<12} {'d_model':<8} {'Experts':<8} {'Layers':<8}")
    print("-" * 80)
    
    for model_name, data in results.items():
        if 'error' in data:
            print(f"{model_name:<35} {'ERROR':<12}")
        else:
            print(f"{model_name:<35} {data['params_millions']:<12.2f} {data['d_model']:<8} {data['num_experts']:<8} {data['n_layers']:<8}")
    
    # Analysis
    print("\n📊 Analysis:")
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if successful_results:
        param_counts = [data['params_millions'] for data in successful_results.values()]
        min_params = min(param_counts)
        max_params = max(param_counts)
        avg_params = sum(param_counts) / len(param_counts)
        
        print(f"   Min Parameters: {min_params:.2f}M")
        print(f"   Max Parameters: {max_params:.2f}M")
        print(f"   Average Parameters: {avg_params:.2f}M")
        print(f"   Range: {max_params/min_params:.2f}x difference")
        
        # Group by category
        print(f"\n📋 By Category:")
        categories = {
            'MLP Scaling': [k for k in successful_results.keys() if k.startswith('mlp_') and 'attention' not in k],
            'Attention+MLP': [k for k in successful_results.keys() if k.startswith('attention_mlp_')],
            'MoE Scaling': [k for k in successful_results.keys() if k.startswith('moe_') and 'attention' not in k],
            'Attention+MoE': [k for k in successful_results.keys() if k.startswith('attention_moe_')],
            'Baseline': [k for k in successful_results.keys() if k == 'baseline']
        }
        
        for category, models in categories.items():
            if models:
                cat_params = [successful_results[m]['params_millions'] for m in models]
                print(f"   {category}: {min(cat_params):.2f}M - {max(cat_params):.2f}M (avg: {sum(cat_params)/len(cat_params):.2f}M)")
    
    return results

if __name__ == "__main__":
    analyze_parameter_counts()
