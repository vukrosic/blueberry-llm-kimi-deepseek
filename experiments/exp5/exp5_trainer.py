"""
Experiment 5: DeepseekV3MoE vs Baseline MoE Comparison Trainer
Fair comparison of MoE implementations with same model size
"""

import torch
import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import setup_muon_optimizer
from utils.helpers import set_seed
from experiments.exp5.exp5_baseline_moe_model import BaselineMoEModel
from experiments.exp5.exp5_simple_deepseek_moe import SimpleDeepseekV3MoEModel


class Experiment5Trainer:
    """Trainer for MoE comparison experiment"""
    
    def __init__(self, base_config: MoEModelConfig, output_dir: str = "experiments/exp5/exp5_results"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data once
        self.texts, self.tokenizer, self.tokens = load_and_cache_data(base_config)
        self.vocab_size = base_config.vocab_size
        
        # Create dataset
        self.dataset = TextTokenDataset(self.tokens, base_config.max_seq_len)
        
        # Train/val split
        val_size = len(self.dataset) // 10
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        print(f"📊 Dataset: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")
        
        # Track results
        self.results = {}
        self.loss_curves = {}
    
    def train_model(self, model, model_name: str) -> Dict[str, Any]:
        """Train a single model with comprehensive metrics"""
        print(f"\n{'='*60}")
        print(f"🧪 Training: {model_name}")
        print(f"{'='*60}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizer
        optimizers = setup_muon_optimizer(model, self.base_config)
        
        # Learning rate schedule
        schedulers = []
        for optimizer in optimizers:
            warmup_steps = self.base_config.max_steps // 20
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (self.base_config.max_steps - warmup_steps)
                    return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            schedulers.append(scheduler)
        
        # Data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.base_config.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.base_config.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        # Training loop
        model.train()
        step = 0
        start_time = time.time()
        
        # Track loss curves
        eval_steps = []
        eval_losses = []
        eval_times = []
        
        print(f"🚀 Starting training for {self.base_config.max_steps} steps...")
        
        while step < self.base_config.max_steps:
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= self.base_config.max_steps:
                    break
                
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                logits, aux_loss = model(x, return_aux_loss=True)
                ce_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.vocab_size), y.view(-1)
                )
                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss
                loss = total_loss / self.base_config.gradient_accumulation_steps
                loss.backward()
                
                # Optimizer step
                if (step + 1) % self.base_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.base_config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                
                # Progress logging
                if step % 100 == 0:
                    print(f"Step {step}/{self.base_config.max_steps}: Loss={ce_loss.item():.4f}")
                
                # Evaluation
                if step % self.base_config.eval_every == 0 and step > 0:
                    print(f"\n🔍 Evaluating model at step {step}...")
                    eval_metrics = self._evaluate_model(model, val_loader)
                    print(f"✅ Val Loss: {eval_metrics['val_loss']:.4f}, Val Acc: {eval_metrics['val_accuracy']:.4f}")
                    
                    # Track for plotting
                    eval_steps.append(step)
                    eval_losses.append(eval_metrics['val_loss'])
                    eval_times.append(time.time())
                
                step += 1
        
        total_time = time.time() - start_time
        
        # Final evaluation
        print(f"\n🔍 Final evaluation...")
        final_eval = self._evaluate_model(model, val_loader)
        
        # Add final evaluation to tracking
        eval_steps.append(self.base_config.max_steps)
        eval_losses.append(final_eval['val_loss'])
        eval_times.append(time.time())
        
        # Store loss curve data
        self.loss_curves[model_name] = {
            'eval_steps': eval_steps,
            'eval_losses': eval_losses,
            'eval_times': eval_times
        }
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            **final_eval,
            'model_name': model_name,
            'training_time_minutes': total_time / 60,
            'parameter_count': param_count,
            'parameters_millions': param_count / 1e6
        }
        
        print(f"✅ {model_name} Results:")
        print(f"   Val Loss: {results['val_loss']:.4f}")
        print(f"   Val Acc: {results['val_accuracy']:.4f}")
        print(f"   Val Perp: {results['val_perplexity']:.2f}")
        print(f"   Time: {results['training_time_minutes']:.2f} min")
        print(f"   Params: {results['parameters_millions']:.2f}M")
        
        return results
    
    def _evaluate_model(self, model, val_loader):
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        # Limit evaluation to first 5 batches for speed
        max_eval_batches = 5
        batch_count = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                if batch_count >= max_eval_batches:
                    break
                    
                x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                
                logits, aux_loss = model(x, return_aux_loss=True)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.vocab_size), y.view(-1)
                )
                
                total_loss += loss.item() * x.size(0) * x.size(1)
                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == y).sum().item()
                total_tokens += y.numel()
                
                batch_count += 1
        
        model.train()
        
        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        perplexity = np.exp(min(avg_loss, 20))
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_perplexity': perplexity
        }
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the full experiment comparing both MoE implementations"""
        print(f"\n🚀 Starting Experiment 5: MoE Implementation Comparison")
        print(f"📋 Comparing: Baseline MoE vs Simple DeepseekV3MoE")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create models
        baseline_model = BaselineMoEModel(self.base_config)
        deepseek_model = SimpleDeepseekV3MoEModel(self.base_config)
        
        # Train both models
        results = {}
        
        # Clear GPU memory before each experiment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        results['baseline'] = self.train_model(baseline_model, 'Baseline MoE')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        results['deepseek'] = self.train_model(deepseek_model, 'Simple DeepseekV3MoE')
        
        # Store results
        self.results = results
        
        # Save results
        self._save_results(results)
        
        # Print comparison
        self._print_comparison(results)
        
        # Create loss vs time plot
        self._create_loss_vs_time_plot()
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results to file"""
        results_file = self.output_dir / "exp5_moe_comparison.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def _print_comparison(self, results: Dict[str, Any]):
        """Print comparison of both MoE implementations"""
        print(f"\n{'='*80}")
        print(f"📊 EXPERIMENT 5 COMPARISON: MoE Implementation Comparison")
        print(f"{'='*80}")
        
        print(f"{'Model':<20} {'Val Loss':<10} {'Val Acc':<10} {'Val Perp':<10} {'Time (min)':<12} {'Params (M)':<12}")
        print(f"{'-'*80}")
        
        for name, result in results.items():
            val_loss = result.get('val_loss', 0)
            val_acc = result.get('val_accuracy', 0)
            val_perp = result.get('val_perplexity', 0)
            time_min = result.get('training_time_minutes', 0)
            params_m = result.get('parameters_millions', 0)
            
            print(f"{name:<20} {val_loss:<10.4f} {val_acc:<10.4f} {val_perp:<10.2f} {time_min:<12.2f} {params_m:<12.2f}")
        
        print(f"{'-'*80}")
        
        # Find best model
        if len(results) >= 2:
            baseline_loss = results['baseline']['val_loss']
            deepseek_loss = results['deepseek']['val_loss']
            
            if deepseek_loss < baseline_loss:
                improvement = (baseline_loss - deepseek_loss) / baseline_loss * 100
                print(f"🏆 DeepseekV3MoE is better by {improvement:.2f}%")
            else:
                improvement = (deepseek_loss - baseline_loss) / deepseek_loss * 100
                print(f"🏆 Baseline MoE is better by {improvement:.2f}%")
        
        print(f"{'='*80}")
    
    def _create_loss_vs_time_plot(self):
        """Create loss vs time visualization on same graph"""
        if not self.loss_curves:
            return
        
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red']
        markers = ['o', 's']
        
        for i, (name, data) in enumerate(self.loss_curves.items()):
            if 'eval_times' in data and len(data['eval_times']) > 1:
                # Convert timestamps to elapsed time in minutes
                start_time = data['eval_times'][0]
                elapsed_times = [(t - start_time) / 60 for t in data['eval_times']]
                
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                plt.plot(elapsed_times, data['eval_losses'], 
                        color=color, marker=marker, linewidth=2, markersize=8,
                        label=f'{name} (Final: {data["eval_losses"][-1]:.4f})')
        
        plt.xlabel('Time (minutes)', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Validation Loss vs Time - Experiment 5: MoE Implementation Comparison', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "loss_vs_time_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Loss vs time comparison plot saved as: {plot_file}")


def main():
    """Main function to run Experiment 5"""
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create configuration for fair comparison
    base_config = MoEModelConfig(
        max_steps=1000,  # Same as exp4 for consistency
        batch_size=16,  # Same batch size
        max_tokens=100000,  # Same dataset size
        eval_every=100,  # Same evaluation frequency
        num_documents=1000,  # Same dataset size
        max_seq_len=256,  # Same sequence length
        d_model=256,  # Same model size
        n_heads=4,    # Same attention heads
        n_layers=3,   # Same number of layers
        d_ff=1024,    # Same feed-forward size
        num_experts=8,  # Same number of experts
        expert_top_k=2,  # Same top-k selection
    )
    
    print(f"🚀 Experiment 5 Configuration:")
    print(f"   Steps: {base_config.max_steps}")
    print(f"   Batch Size: {base_config.batch_size}")
    print(f"   Model: {base_config.d_model}d, {base_config.n_layers}L, {base_config.n_heads}H")
    print(f"   MoE: {base_config.num_experts} experts, top-{base_config.expert_top_k}")
    print(f"   Expected Training Time: ~5-10 minutes per model")
    
    # Create trainer
    trainer = Experiment5Trainer(base_config)
    
    # Run experiment
    results = trainer.run_experiment()
    
    print(f"\n✅ Experiment 5 completed!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
