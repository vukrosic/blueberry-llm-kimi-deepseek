"""
Experiment 9: DeepSeek Attention + MLP Training
Focused on training the DeepSeek Attention + MLP model with learning rate optimization
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

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import setup_muon_optimizer
from utils.helpers import set_seed
from experiments.exp1_simplified_ablation_study.exp1_models import AttentionMLP_512dModel
from benchmark_evaluator import HellaSwagEvaluator


class LongTermExperiment2Trainer:
    """Long-term trainer for Attention+MLP 512d model"""
    
    def __init__(self, base_config: MoEModelConfig, output_dir: str = "exp2_results"):
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
        self.checkpoints = {}
        
        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize HellaSwag evaluator
        self.hellaswag_evaluator = HellaSwagEvaluator(output_dir=str(self.output_dir / "hellaswag_benchmark"))
    
    def train_model_long_term(self, model, model_name: str, total_steps: int = 10000, 
                            checkpoint_every: int = 3000, eval_every: int = 100, 
                            hellaswag_every: int = 1000) -> Dict[str, Any]:
        """Train model for extended period with regular checkpoints"""
        print(f"\n{'='*80}")
        print(f"🚀 Long-term Training: {model_name} ({total_steps} steps)")
        print(f"📁 Checkpoints every {checkpoint_every} steps")
        print(f"📊 Evaluation every {eval_every} steps")
        print(f"🧪 HellaSwag benchmark every {hellaswag_every} steps")
        print(f"{'='*80}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizer
        optimizers = setup_muon_optimizer(model, self.base_config)
        
        # Learning rate schedule with longer warmup
        schedulers = []
        for optimizer in optimizers:
            warmup_steps = max(100, total_steps // 20)  # 5% warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
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
        train_losses = []
        
        print(f"🚀 Starting long-term training for {total_steps} steps...")
        
        while step < total_steps:
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= total_steps:
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
                
                # Progress logging (every eval_every steps)
                if step % eval_every == 0:
                    print(f"Step {step}/{total_steps}: Loss={ce_loss.item():.6f}")
                    train_losses.append(ce_loss.item())
                
                # Evaluation (every eval_every steps)
                if step % eval_every == 0 and step > 0:
                    eval_metrics = self._evaluate_model(model, val_loader)
                    
                    # Track for plotting
                    eval_steps.append(step)
                    eval_losses.append(eval_metrics['val_loss'])
                    eval_times.append(time.time())
                    
                    print(f"   Val Loss: {eval_metrics['val_loss']:.6f}, Val Acc: {eval_metrics['val_accuracy']:.4f}")
                
                # Checkpoint saving (every checkpoint_every steps)
                if step % checkpoint_every == 0 and step > 0:
                    self._save_checkpoint(model, step, eval_losses[-1] if eval_losses else 0)
                
                # HellaSwag benchmark evaluation (every hellaswag_every steps)
                if step % hellaswag_every == 0 and step > 0:
                    print(f"\n🧪 Running HellaSwag benchmark at step {step}...")
                    try:
                        benchmark_result = self.hellaswag_evaluator.evaluate_model(model, model_name)
                        print(f"✅ HellaSwag benchmark completed: {benchmark_result}")
                    except Exception as e:
                        print(f"❌ HellaSwag benchmark failed: {e}")
                
                step += 1
        
        total_time = time.time() - start_time
        
        # Final evaluation
        print(f"\n🔍 Final evaluation...")
        final_eval = self._evaluate_model(model, val_loader)
        
        # Add final evaluation to tracking
        eval_steps.append(total_steps)
        eval_losses.append(final_eval['val_loss'])
        eval_times.append(time.time())
        
        # Store loss curve data
        self.loss_curves[model_name] = {
            'eval_steps': eval_steps,
            'eval_losses': eval_losses,
            'eval_times': eval_times,
            'train_losses': train_losses
        }
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            **final_eval,
            'model_name': model_name,
            'training_time_minutes': total_time / 60,
            'parameter_count': param_count,
            'parameters_millions': param_count / 1e6,
            'total_steps': total_steps,
            'checkpoint_every': checkpoint_every,
            'eval_every': eval_every
        }
        
        print(f"✅ {model_name} Long-term Training Results:")
        print(f"   Val Loss: {results['val_loss']:.6f}")
        print(f"   Val Acc: {results['val_accuracy']:.6f}")
        print(f"   Val Perp: {results['val_perplexity']:.4f}")
        print(f"   Time: {results['training_time_minutes']:.2f} min")
        print(f"   Params: {results['parameters_millions']:.2f}M")
        print(f"   Steps: {results['total_steps']}")
        
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
    
    def _save_checkpoint(self, model, step: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Keep track of checkpoints
        self.checkpoints[step] = {
            'path': str(checkpoint_path),
            'val_loss': val_loss,
            'timestamp': checkpoint['timestamp']
        }
    
    def run_long_term_training(self, total_steps: int = 10000, 
                              checkpoint_every: int = 3000, eval_every: int = 100,
                              hellaswag_every: int = 1000) -> Dict[str, Any]:
        """Run long-term training experiment"""
        print(f"\n🚀 Starting Long-term Experiment 9: Attention+MLP 512d")
        print(f"📋 Training for {total_steps} steps")
        print(f"💾 Checkpoints every {checkpoint_every} steps")
        print(f"📊 Evaluation every {eval_every} steps")
        print(f"🧪 HellaSwag benchmark every {hellaswag_every} steps")
        
        # Set seed for reproducibility
        set_seed(42)
        
        try:
            print(f"\n🧪 Creating Attention+MLP 512d model...")
            
            # Create model
            model = AttentionMLP_512dModel(self.base_config)
            
            # Train for extended period
            result = self.train_model_long_term(
                model, 
                "attention_mlp_512d", 
                total_steps=total_steps,
                checkpoint_every=checkpoint_every,
                eval_every=eval_every,
                hellaswag_every=hellaswag_every
            )
            
            self.results["attention_mlp_512d"] = result
            
            # Save results
            self._save_results(result, "long_term")
            
            # Create training visualization
            self._create_training_plot()
            
            print(f"\n📊 Long-term Training Summary:")
            print(f"   ✅ Completed: {total_steps} steps")
            print(f"   💾 Checkpoints: {len(self.checkpoints)} saved")
            print(f"   📁 Results saved in: {self.output_dir}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in long-term training: {e}")
            return {"error": str(e), "model_name": "attention_mlp_512d"}
    
    def _save_results(self, results: Dict[str, Any], mode: str):
        """Save experiment results to file"""
        results_file = self.output_dir / f"exp2_{mode}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def _create_training_plot(self):
        """Create training loss visualization"""
        if not self.loss_curves:
            return
        
        model_name = "attention_mlp_512d"
        if model_name not in self.loss_curves:
            return
        
        data = self.loss_curves[model_name]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Validation Loss vs Steps
        if 'eval_steps' in data and 'eval_losses' in data:
            ax1.plot(data['eval_steps'], data['eval_losses'], 
                    'b-', linewidth=2, marker='o', markersize=4, alpha=0.8)
            ax1.set_xlabel('Training Steps', fontsize=12)
            ax1.set_ylabel('Validation Loss', fontsize=12)
            ax1.set_title(f'Long-term Training: Validation Loss vs Steps ({model_name})', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')  # Log scale for better visualization
        
        # Plot 2: Training Loss vs Steps
        if 'train_losses' in data:
            train_steps = list(range(0, len(data['train_losses']) * 100, 100))
            ax2.plot(train_steps[:len(data['train_losses'])], data['train_losses'], 
                    'r-', linewidth=1, alpha=0.7)
            ax2.set_xlabel('Training Steps', fontsize=12)
            ax2.set_ylabel('Training Loss', fontsize=12)
            ax2.set_title(f'Long-term Training: Training Loss vs Steps ({model_name})', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "exp2_long_term_training_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Training visualization created: {plot_file}")


def main():
    """Main function to run Long-term Experiment 9"""
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create configuration for long-term training with optimal learning rate
    base_config = MoEModelConfig(
        max_steps=10000,  # Will be overridden by total_steps parameter
        batch_size=16,   # Further reduced batch size for memory
        max_tokens=100000,  # Full dataset
        eval_every=100,  # Evaluation every 100 steps
        num_documents=1000,  # Full dataset
        max_seq_len=256,
        d_model=512,  # Target 512 scale
        n_heads=8,    # Powers of 2
        n_layers=6,   # Reduced layers for memory constraints
        d_ff=2048,    # Powers of 2 (4x d_model)
        num_experts=8,  # Powers of 2
        expert_top_k=2,  # Keep same top-k
        muon_lr=3e-3,  # Optimal learning rate from LR search
    )
    
    print(f"🚀 Long-term Experiment 9 Configuration:")
    print(f"   Model: Attention+MLP 512d (best from Exp8)")
    print(f"   Batch Size: {base_config.batch_size}")
    print(f"   Model: {base_config.d_model}d, {base_config.n_layers}L, {base_config.n_heads}H")
    print(f"   Training: Extended long-term training")
    print(f"   Checkpoints: Regular saves during training")
    
    # Create trainer
    trainer = LongTermExperiment2Trainer(base_config)
    
    # Run long-term training experiment
    print(f"\n🧪 Running long-term training experiment...")
    
    # Run with configurable parameters
    results = trainer.run_long_term_training(
        total_steps=10000,      # Train for 10k steps
        checkpoint_every=3000,   # Save checkpoint every 3k steps
        eval_every=100,         # Evaluate every 100 steps
        hellaswag_every=1000    # HellaSwag benchmark every 1k steps
    )
    
    print(f"\n✅ Long-term Experiment 9 completed!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
