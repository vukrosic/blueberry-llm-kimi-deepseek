"""
Experiment 3: Advanced DeepSeek Attention Features

This experiment tests advanced DeepSeek attention mechanisms that weren't covered
in experiments 1 and 2. Focuses on sophisticated attention configurations.

Features:
- Q-LoRA and KV-LoRA attention projections
- Advanced RoPE scaling and head dimension variants
- Flash Attention 2 implementation
- Mixed head dimensions (different Q, K, V sizes)
- Attention bias configurations
- Advanced MoE integration patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from scipy import stats
import torch.profiler
from contextlib import nullcontext
from itertools import product

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed
from experiments.exp3.exp3_config_import import (
    get_advanced_deepseek_configs, 
    get_training_configs,
    create_moe_config_from_deepseek,
    create_config_from_moe_config
)
from experiments.exp1.exp1_deepseek_import import DeepSeekMoEModel
from models.moe_llm import MoEMinimalLLM


# Import MixtureOfExperts from the correct location
from models.layers import MixtureOfExperts


def run_advanced_deepseek_experiment():
    """Run the Advanced DeepSeek experiment"""
    print("🚀 Starting Experiment 3: Advanced DeepSeek Attention Features")
    print("=" * 60)
    
    # Set random seed
    set_seed(42)
    
    # Get configurations
    deepseek_configs = get_advanced_deepseek_configs()
    training_configs = get_training_configs()
    
    results = []
    
    for i, (deepseek_config, training_config) in enumerate(zip(deepseek_configs, training_configs)):
        print(f"\n📊 Configuration {i+1}/{len(deepseek_configs)}")
        print(f"Model: {deepseek_config['name']}")
        print(f"Parameters: {deepseek_config['params']}")
        
        # Create MoE config
        moe_config = create_moe_config_from_deepseek(deepseek_config)
        
        # Create model
        model = DeepSeekMoEModel(moe_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Load data
        print("📚 Loading data...")
        train_data, val_data = load_and_cache_data()
        
        # Create datasets
        train_dataset = TextTokenDataset(train_data, moe_config.max_seq_len)
        val_dataset = TextTokenDataset(val_data, moe_config.max_seq_len)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_config['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=training_config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # Training loop
        print("🏋️ Starting training...")
        model.train()
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        train_losses = []
        val_losses = []
        
        for epoch in range(training_config['num_epochs']):
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids, labels=labels)
                loss = outputs['loss']
                
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, labels=labels)
                    val_loss += outputs['loss'].item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            model.train()
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        training_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Store results
        result = {
            'config_name': mamba_config['name'],
            'model_params': mamba_config['params'],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'training_time': training_time,
            'memory_used_mb': memory_used,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'num_epochs': training_config['num_epochs'],
            'batch_size': training_config['batch_size'],
            'learning_rate': training_config['learning_rate']
        }
        
        results.append(result)
        
        print(f"✅ Configuration {i+1} completed")
        print(f"Training time: {training_time:.2f}s")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final val loss: {val_losses[-1]:.4f}")
    
    # Save results
    results_dir = Path("experiments/exp3/exp3_advanced_deepseek_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "advanced_deepseek_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    create_advanced_deepseek_visualization(results, results_dir)
    
    print("\n🎉 Experiment 3 completed!")
    print(f"Results saved to: {results_dir}")
    
    return results


def create_advanced_deepseek_visualization(results, results_dir):
    """Create visualization for Advanced DeepSeek experiment results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment 3: Advanced DeepSeek Attention Features Results', fontsize=16, fontweight='bold')
    
    # Extract data
    config_names = [r['config_name'] for r in results]
    final_train_losses = [r['final_train_loss'] for r in results]
    final_val_losses = [r['final_val_loss'] for r in results]
    training_times = [r['training_time'] for r in results]
    memory_usage = [r['memory_used_mb'] for r in results]
    
    # Plot 1: Final losses
    x = np.arange(len(config_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, final_train_losses, width, label='Train Loss', alpha=0.8)
    axes[0, 0].bar(x + width/2, final_val_losses, width, label='Val Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Configuration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Final Training and Validation Losses')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(config_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training time
    axes[0, 1].bar(config_names, training_times, alpha=0.8, color='green')
    axes[0, 1].set_xlabel('Configuration')
    axes[0, 1].set_ylabel('Training Time (seconds)')
    axes[0, 1].set_title('Training Time Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Memory usage
    axes[1, 0].bar(config_names, memory_usage, alpha=0.8, color='orange')
    axes[1, 0].set_xlabel('Configuration')
    axes[1, 0].set_ylabel('Memory Usage (MB)')
    axes[1, 0].set_title('Memory Usage Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss curves
    for i, result in enumerate(results):
        epochs = range(1, len(result['train_losses']) + 1)
        axes[1, 1].plot(epochs, result['train_losses'], label=f"{result['config_name']} (Train)", linestyle='-')
        axes[1, 1].plot(epochs, result['val_losses'], label=f"{result['config_name']} (Val)", linestyle='--')
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training Progress')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "advanced_deepseek_experiment_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved to: {results_dir / 'advanced_deepseek_experiment_comprehensive.png'}")


if __name__ == "__main__":
    results = run_advanced_deepseek_experiment()
