"""
Experiment 3: Mamba State Space Model (No Attention)

This experiment tests the Mamba architecture as an alternative to attention mechanisms.
Mamba uses Structured State Space Models (SSMs) to process sequences efficiently
without relying on attention mechanisms.

Features:
- Mamba SSM blocks instead of attention
- State space modeling for sequence processing
- Efficient long sequence handling
- Comparison with attention-based models
- MoE integration with Mamba blocks
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
    get_mamba_configs, 
    get_training_configs,
    create_moe_config_from_mamba,
    create_config_from_moe_config
)
from models.moe_llm import MoEMinimalLLM


class MambaSSM(nn.Module):
    """
    Simplified Mamba State Space Model block
    This is a basic implementation for experimentation
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # State space parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, d_state, bias=True)
        
        # State space matrices (simplified)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)
        
        # Convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # (B, d_inner, L)
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # State space parameters
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state)
        delta, B_param = x_dbl.chunk(2, dim=-1)  # (B, L, d_state)
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_state)
        
        # Simplified state space computation
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretization (simplified)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B_param.unsqueeze(-2) * x.unsqueeze(-1)  # (B, L, d_inner, d_state)
        
        # State space recurrence (simplified)
        y = torch.zeros_like(x)  # (B, L, d_inner)
        for i in range(L):
            if i == 0:
                y[:, i] = deltaB_u[:, i].sum(-1)
            else:
                y[:, i] = (deltaA[:, i] * y[:, i-1].unsqueeze(-1)).sum(-1) + deltaB_u[:, i].sum(-1)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        return output


class MambaMoEBlock(nn.Module):
    """Mamba block with MoE feed-forward"""
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()
        
        # Mamba SSM layer
        self.mamba = MambaSSM(d_model, d_state, d_conv, expand)
        
        # MoE layer
        self.feed_forward = MixtureOfExperts(
            d_model, d_ff, num_experts, top_k, dropout
        )
        
        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Mamba SSM
        mamba_out = self.mamba(self.norm1(x))
        x = x + self.dropout(mamba_out)
        
        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss


class MambaMoEModel(nn.Module):
    """Mamba-based MoE model without attention"""
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Mamba MoE blocks
        self.blocks = nn.ModuleList([
            MambaMoEBlock(
                d_model=config.d_model,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                num_experts=config.num_experts,
                top_k=config.top_k,
                dropout=config.dropout,
                d_state=getattr(config, 'd_state', 16),
                d_conv=getattr(config, 'd_conv', 4),
                expand=getattr(config, 'expand', 2)
            )
            for _ in range(config.n_layers)
        ])
        
        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        
        # Forward through Mamba blocks
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss
        
        # Final normalization and output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            # Add auxiliary loss
            if total_aux_loss > 0:
                loss += 0.01 * total_aux_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'aux_loss': total_aux_loss
        }


def run_mamba_experiment():
    """Run the Mamba experiment"""
    print("🚀 Starting Experiment 3: Mamba State Space Model")
    print("=" * 60)
    
    # Set random seed
    set_seed(42)
    
    # Get configurations
    mamba_configs = get_mamba_configs()
    training_configs = get_training_configs()
    
    results = []
    
    for i, (mamba_config, training_config) in enumerate(zip(mamba_configs, training_configs)):
        print(f"\n📊 Configuration {i+1}/{len(mamba_configs)}")
        print(f"Model: {mamba_config['name']}")
        print(f"Parameters: {mamba_config['params']}")
        
        # Create MoE config
        moe_config = create_moe_config_from_mamba(mamba_config)
        
        # Create model
        model = MambaMoEModel(moe_config)
        
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
    results_dir = Path("experiments/exp3/exp3_mamba_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "mamba_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    create_mamba_visualization(results, results_dir)
    
    print("\n🎉 Experiment 3 completed!")
    print(f"Results saved to: {results_dir}")
    
    return results


def create_mamba_visualization(results, results_dir):
    """Create visualization for Mamba experiment results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment 3: Mamba State Space Model Results', fontsize=16, fontweight='bold')
    
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
    plt.savefig(results_dir / "mamba_experiment_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved to: {results_dir / 'mamba_experiment_comprehensive.png'}")


if __name__ == "__main__":
    results = run_mamba_experiment()
