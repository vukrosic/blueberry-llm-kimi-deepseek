"""
Learning Rate Search for Experiment 9
Systematic search across different learning rates to find optimal training parameters
"""

import torch
import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple
from pathlib import Path
import itertools

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import setup_muon_optimizer
from utils.helpers import set_seed
from experiments.exp8.exp8_reduced_ablation_models import AttentionMLP_512dModel


class LearningRateSearchTrainer:
    """Learning rate search trainer for Experiment 9"""
    
    def __init__(self, base_config: MoEModelConfig, output_dir: str = "exp9_lr_search"):
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
        self.lr_results = {}
        self.loss_curves = {}
        
        # Create results directory
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def train_with_lr(self, model, lr: float, model_name: str, 
                     max_steps: int = 2000, eval_every: int = 100,
                     early_stopping_patience: int = 5) -> Dict[str, Any]:
        """Train model with specific learning rate"""
        print(f"\n{'='*60}")
        print(f"🚀 Training with LR: {lr:.2e}")
        print(f"📋 Max steps: {max_steps}")
        print(f"📊 Eval every: {eval_every} steps")
        print(f"⏹️  Early stopping patience: {early_stopping_patience}")
        print(f"{'='*60}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizer with specific learning rate
        optimizers = setup_muon_optimizer(model, self.base_config)
        
        # Override learning rates for all optimizers
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Learning rate schedule with warmup
        schedulers = []
        for optimizer in optimizers:
            warmup_steps = max(50, max_steps // 20)  # 5% warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (max_steps - warmup_steps)
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
        
        # Track metrics
        eval_steps = []
        eval_losses = []
        train_losses = []
        learning_rates = []
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_step = 0
        
        print(f"🚀 Starting training with LR {lr:.2e}...")
        
        while step < max_steps:
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= max_steps:
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
                
                # Track learning rate
                current_lr = optimizers[0].param_groups[0]['lr']
                learning_rates.append(current_lr)
                
                # Progress logging
                if step % eval_every == 0:
                    print(f"Step {step}/{max_steps}: Loss={ce_loss.item():.6f}, LR={current_lr:.2e}")
                    train_losses.append(ce_loss.item())
                
                # Evaluation
                if step % eval_every == 0 and step > 0:
                    eval_metrics = self._evaluate_model(model, val_loader)
                    
                    # Track for plotting
                    eval_steps.append(step)
                    eval_losses.append(eval_metrics['val_loss'])
                    
                    print(f"   Val Loss: {eval_metrics['val_loss']:.6f}, Val Acc: {eval_metrics['val_accuracy']:.4f}")
                    
                    # Early stopping check
                    if eval_metrics['val_loss'] < best_val_loss:
                        best_val_loss = eval_metrics['val_loss']
                        best_step = step
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Early stopping
                    if patience_counter >= early_stopping_patience:
                        print(f"⏹️  Early stopping at step {step} (patience: {patience_counter})")
                        break
                
                step += 1
        
        total_time = time.time() - start_time
        
        # Final evaluation
        print(f"\n🔍 Final evaluation...")
        final_eval = self._evaluate_model(model, val_loader)
        
        # Add final evaluation to tracking
        eval_steps.append(step)
        eval_losses.append(final_eval['val_loss'])
        
        # Store loss curve data
        self.loss_curves[f"lr_{lr:.2e}"] = {
            'eval_steps': eval_steps,
            'eval_losses': eval_losses,
            'train_losses': train_losses,
            'learning_rates': learning_rates
        }
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            **final_eval,
            'learning_rate': lr,
            'model_name': model_name,
            'training_time_minutes': total_time / 60,
            'parameter_count': param_count,
            'parameters_millions': param_count / 1e6,
            'total_steps': step,
            'best_val_loss': best_val_loss,
            'best_step': best_step,
            'early_stopped': patience_counter >= early_stopping_patience,
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_lr': learning_rates[-1] if learning_rates else lr
        }
        
        print(f"✅ LR {lr:.2e} Training Results:")
        print(f"   Val Loss: {results['val_loss']:.6f}")
        print(f"   Val Acc: {results['val_accuracy']:.6f}")
        print(f"   Val Perp: {results['val_perplexity']:.4f}")
        print(f"   Time: {results['training_time_minutes']:.2f} min")
        print(f"   Steps: {results['total_steps']}")
        print(f"   Best Val Loss: {results['best_val_loss']:.6f} at step {results['best_step']}")
        print(f"   Early Stopped: {results['early_stopped']}")
        
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
    
    def run_lr_search(self, learning_rates: List[float], max_steps: int = 2000,
                     eval_every: int = 100, early_stopping_patience: int = 5) -> Dict[str, Any]:
        """Run learning rate search across multiple learning rates"""
        print(f"\n🚀 Starting Learning Rate Search for Experiment 9")
        print(f"📋 Learning rates to test: {learning_rates}")
        print(f"📊 Max steps per LR: {max_steps}")
        print(f"⏹️  Early stopping patience: {early_stopping_patience}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        search_results = {}
        
        for i, lr in enumerate(learning_rates):
            print(f"\n{'='*80}")
            print(f"🧪 Learning Rate Search {i+1}/{len(learning_rates)}: LR = {lr:.2e}")
            print(f"{'='*80}")
            
            try:
                # Create fresh model for each learning rate
                model = AttentionMLP_512dModel(self.base_config)
                
                # Train with this learning rate
                result = self.train_with_lr(
                    model, 
                    lr, 
                    f"attention_mlp_512d_lr_{lr:.2e}",
                    max_steps=max_steps,
                    eval_every=eval_every,
                    early_stopping_patience=early_stopping_patience
                )
                
                search_results[f"lr_{lr:.2e}"] = result
                
                # Save individual result
                self._save_lr_result(result, lr)
                
                print(f"✅ Completed LR {lr:.2e}")
                
            except Exception as e:
                print(f"❌ Error with LR {lr:.2e}: {e}")
                search_results[f"lr_{lr:.2e}"] = {"error": str(e), "learning_rate": lr}
        
        # Save all results
        self._save_search_results(search_results)
        
        # Create comparison plots
        self._create_lr_comparison_plots()
        
        # Find best learning rate
        best_lr = self._find_best_learning_rate(search_results)
        
        print(f"\n📊 Learning Rate Search Summary:")
        print(f"   ✅ Tested {len(learning_rates)} learning rates")
        print(f"   🏆 Best LR: {best_lr['learning_rate']:.2e}")
        print(f"   📈 Best Val Loss: {best_lr['val_loss']:.6f}")
        print(f"   📁 Results saved in: {self.output_dir}")
        
        return search_results
    
    def _save_lr_result(self, result: Dict[str, Any], lr: float):
        """Save individual learning rate result"""
        result_file = self.results_dir / f"lr_{lr:.2e}_result.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 LR {lr:.2e} result saved to: {result_file}")
    
    def _save_search_results(self, search_results: Dict[str, Any]):
        """Save all learning rate search results"""
        results_file = self.output_dir / "lr_search_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(search_results, f, indent=2)
        
        print(f"💾 All LR search results saved to: {results_file}")
    
    def _find_best_learning_rate(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best learning rate based on validation loss"""
        valid_results = {k: v for k, v in search_results.items() if 'error' not in v}
        
        if not valid_results:
            return {"error": "No valid results found"}
        
        best_lr_key = min(valid_results.keys(), key=lambda k: valid_results[k]['val_loss'])
        best_result = valid_results[best_lr_key]
        
        print(f"\n🏆 Best Learning Rate Analysis:")
        print(f"   Best LR: {best_result['learning_rate']:.2e}")
        print(f"   Val Loss: {best_result['val_loss']:.6f}")
        print(f"   Val Acc: {best_result['val_accuracy']:.6f}")
        print(f"   Val Perp: {best_result['val_perplexity']:.4f}")
        print(f"   Training Time: {best_result['training_time_minutes']:.2f} min")
        print(f"   Steps: {best_result['total_steps']}")
        
        return best_result
    
    def _create_lr_comparison_plots(self):
        """Create comparison plots for different learning rates"""
        if not self.loss_curves:
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot 1: Validation Loss vs Steps for all LRs
        for lr_key, data in self.loss_curves.items():
            if 'eval_steps' in data and 'eval_losses' in data:
                lr_value = float(lr_key.split('_')[1])
                ax1.plot(data['eval_steps'], data['eval_losses'], 
                        linewidth=2, marker='o', markersize=3, alpha=0.8,
                        label=f'LR = {lr_value:.2e}')
        
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Validation Loss', fontsize=12)
        ax1.set_title('Learning Rate Search: Validation Loss vs Steps', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.legend()
        
        # Plot 2: Final Validation Loss vs Learning Rate
        lr_values = []
        final_losses = []
        for lr_key, data in self.loss_curves.items():
            if 'eval_losses' in data and data['eval_losses']:
                lr_value = float(lr_key.split('_')[1])
                lr_values.append(lr_value)
                final_losses.append(data['eval_losses'][-1])
        
        if lr_values and final_losses:
            ax2.semilogx(lr_values, final_losses, 'bo-', linewidth=2, markersize=8)
            ax2.set_xlabel('Learning Rate', fontsize=12)
            ax2.set_ylabel('Final Validation Loss', fontsize=12)
            ax2.set_title('Final Validation Loss vs Learning Rate', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training Loss vs Steps for all LRs
        for lr_key, data in self.loss_curves.items():
            if 'train_losses' in data:
                lr_value = float(lr_key.split('_')[1])
                train_steps = list(range(0, len(data['train_losses']) * 100, 100))
                ax3.plot(train_steps[:len(data['train_losses'])], data['train_losses'], 
                        linewidth=1, alpha=0.7, label=f'LR = {lr_value:.2e}')
        
        ax3.set_xlabel('Training Steps', fontsize=12)
        ax3.set_ylabel('Training Loss', fontsize=12)
        ax3.set_title('Learning Rate Search: Training Loss vs Steps', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        ax3.legend()
        
        # Plot 4: Learning Rate Schedule
        for lr_key, data in self.loss_curves.items():
            if 'learning_rates' in data:
                lr_value = float(lr_key.split('_')[1])
                steps = list(range(len(data['learning_rates'])))
                ax4.plot(steps, data['learning_rates'], 
                        linewidth=2, alpha=0.8, label=f'LR = {lr_value:.2e}')
        
        ax4.set_xlabel('Training Steps', fontsize=12)
        ax4.set_ylabel('Learning Rate', fontsize=12)
        ax4.set_title('Learning Rate Schedule', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "lr_search_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Learning rate comparison plots created: {plot_file}")


def main():
    """Main function to run Learning Rate Search for Experiment 9"""
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create configuration for learning rate search
    base_config = MoEModelConfig(
        max_steps=2000,  # Shorter training for LR search
        batch_size=128,
        max_tokens=100000,
        eval_every=100,
        num_documents=1000,
        max_seq_len=256,
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        num_experts=8,
        expert_top_k=2,
    )
    
    print(f"🚀 Learning Rate Search Configuration:")
    print(f"   Model: Attention+MLP 512d")
    print(f"   Batch Size: {base_config.batch_size}")
    print(f"   Model: {base_config.d_model}d, {base_config.n_layers}L, {base_config.n_heads}H")
    print(f"   Training: Learning rate search")
    
    # Define learning rates to test
    learning_rates = [
        1e-5,   # Very low
        3e-5,   # Low
        1e-4,   # Medium-low
        3e-4,   # Medium
        1e-3,   # Medium-high
        3e-3,   # High
        1e-2,   # Very high
    ]
    
    print(f"📋 Learning rates to test: {learning_rates}")
    
    # Create trainer
    trainer = LearningRateSearchTrainer(base_config)
    
    # Run learning rate search
    print(f"\n🧪 Running learning rate search...")
    
    results = trainer.run_lr_search(
        learning_rates=learning_rates,
        max_steps=2000,              # 2k steps per LR
        eval_every=100,              # Evaluate every 100 steps
        early_stopping_patience=5    # Early stop after 5 evaluations without improvement
    )
    
    print(f"\n✅ Learning Rate Search completed!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
