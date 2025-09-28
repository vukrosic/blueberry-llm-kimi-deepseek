#!/usr/bin/env python3
"""
Extended Training for Experiment 10: DeepSeek Attention + GLM4 MoE
Focused on training the DeepSeek Attention + GLM4 MoE model with optimal configurations
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
from experiments.exp1_simplified_ablation_study.exp1_models import MoEMinimalLLM
from benchmark_evaluator import HellaSwagEvaluator


class ExtendedExperiment3Trainer:
    """Extended trainer for DeepSeek Attention + GLM4 MoE model"""
    
    def __init__(self, config: MoEModelConfig, output_dir: str = "exp3_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        print("📚 Loading data...")
        self.texts, self.tokenizer, self.tokens = load_and_cache_data(config)
        
        # Create dataset
        self.dataset = TextTokenDataset(self.tokens, config.max_seq_len)
        
        # Train/val split
        val_size = len(self.dataset) // 10
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        print(f"✅ Data loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")
        
        # Initialize HellaSwag evaluator
        self.hellaswag_evaluator = HellaSwagEvaluator()
    
    def train_step(self, model, optimizers, batch):
        """Single training step"""
        x, y = batch
        input_ids = x.cuda() if torch.cuda.is_available() else x
        labels = y.cuda() if torch.cuda.is_available() else y
        
        for optimizer in optimizers:
            optimizer.zero_grad()
        outputs, aux_loss = model(input_ids, return_aux_loss=True)
        
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)), 
            labels.view(-1)
        )
        
        # Add auxiliary loss if present
        if aux_loss is not None:
            loss = loss + aux_loss
        
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        
        return loss.item()
    
    def evaluate(self, model, val_loader):
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                input_ids = x.cuda() if torch.cuda.is_available() else x
                labels = y.cuda() if torch.cuda.is_available() else y
                
                outputs, aux_loss = model(input_ids, return_aux_loss=True)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    labels.view(-1)
                )
                
                # Add auxiliary loss if present
                if aux_loss is not None:
                    loss = loss + aux_loss
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=-1)
                total_correct += (predictions == labels).sum().item()
                total_tokens += labels.numel()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        model.train()
        return avg_loss, accuracy, perplexity
    
    def run_extended_training(self, 
                            total_steps: int = 10000,
                            checkpoint_every: int = 3000,
                            eval_every: int = 100,
                            hellaswag_every: int = 1000,
                            learning_rate: float = 1e-3) -> Dict[str, Any]:
        """Run extended training with checkpoints and evaluation"""
        
        print(f"🚀 Starting Extended Training for DeepSeek Attention + GLM4 MoE")
        print(f"📋 Configuration:")
        print(f"   Model: DeepSeek Attention + GLM4 MoE 512d")
        print(f"   Experts: {self.config.num_experts}, Top-k: {self.config.expert_top_k}")
        print(f"   Total Steps: {total_steps}")
        print(f"   Checkpoint Every: {checkpoint_every} steps")
        print(f"   Evaluation Every: {eval_every} steps")
        print(f"   HellaSwag Every: {hellaswag_every} steps")
        print(f"   Learning Rate: {learning_rate:.2e}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create model
        model = MoEMinimalLLM(self.config)
        model = model.cuda() if torch.cuda.is_available() else model
        
        # Create data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Setup optimizer with custom learning rate
        # Temporarily modify config for this training run
        original_lr = self.config.muon_lr
        self.config.muon_lr = learning_rate
        optimizers = setup_muon_optimizer(model, self.config)
        self.config.muon_lr = original_lr  # Restore original
        
        # Training tracking
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_perplexities = []
        step_times = []
        checkpoint_info = []
        hellaswag_results = {}
        
        step = 0
        start_time = time.time()
        
        print(f"\n🧪 Starting training...")
        
        while step < total_steps:
            for batch in train_loader:
                if step >= total_steps:
                    break
                
                # Training step
                step_start = time.time()
                loss = self.train_step(model, optimizers, batch)
                step_time = time.time() - step_start
                
                train_losses.append(loss)
                step_times.append(step_time)
                step += 1
                
                # Print progress
                if step % 50 == 0:
                    avg_time = np.mean(step_times[-50:])
                    print(f"Step {step}/{total_steps}: Loss={loss:.4f}, Time={avg_time:.3f}s/step")
                
                # Evaluation
                if step % eval_every == 0:
                    val_loss, val_acc, val_perp = self.evaluate(model, val_loader)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)
                    val_perplexities.append(val_perp)
                    
                    print(f"  Step {step}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val Perp={val_perp:.4f}")
                
                # Checkpoint saving
                if step % checkpoint_every == 0:
                    checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dicts': [opt.state_dict() for opt in optimizers],
                        'val_loss': val_losses[-1] if val_losses else None,
                        'val_accuracy': val_accuracies[-1] if val_accuracies else None,
                        'timestamp': time.time()
                    }, checkpoint_path)
                    
                    checkpoint_info.append({
                        'step': step,
                        'val_loss': val_losses[-1] if val_losses else None,
                        'val_accuracy': val_accuracies[-1] if val_accuracies else None,
                        'timestamp': time.time()
                    })
                    
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                
                # HellaSwag benchmark
                if step % hellaswag_every == 0:
                    print(f"🎯 Running HellaSwag benchmark at step {step}...")
                    try:
                        hellaswag_result = self.hellaswag_evaluator.evaluate_model(model)
                        hellaswag_results[f"step_{step}"] = hellaswag_result
                        
                        # Save HellaSwag results
                        hellaswag_file = self.output_dir / "hellaswag_benchmark" / f"step_{step}_hellaswag_results.json"
                        hellaswag_file.parent.mkdir(exist_ok=True)
                        with open(hellaswag_file, 'w') as f:
                            json.dump(hellaswag_result, f, indent=2)
                        
                        print(f"📊 HellaSwag Score: {hellaswag_result.get('accuracy', 'N/A')}")
                        
                    except Exception as e:
                        print(f"❌ HellaSwag benchmark failed: {e}")
                        hellaswag_results[f"step_{step}"] = {'error': str(e)}
        
        total_time = time.time() - start_time
        
        # Final evaluation
        final_val_loss, final_val_acc, final_val_perp = self.evaluate(model, val_loader)
        
        # Save final model
        final_model_path = self.output_dir / "final_model.pt"
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dicts': [opt.state_dict() for opt in optimizers],
            'final_val_loss': final_val_loss,
            'final_val_accuracy': final_val_acc,
            'final_val_perplexity': final_val_perp,
            'total_training_time': total_time,
            'timestamp': time.time()
        }, final_model_path)
        
        print(f"💾 Final model saved: {final_model_path}")
        
        # Create training curves
        self.create_training_curves(train_losses, val_losses, val_accuracies, val_perplexities)
        
        # Compile results
        results = {
            'model_config': {
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'd_ff': self.config.d_ff,
                'num_experts': self.config.num_experts,
                'expert_top_k': self.config.expert_top_k,
            },
            'training_config': {
                'total_steps': total_steps,
                'learning_rate': learning_rate,
                'batch_size': self.config.batch_size,
                'checkpoint_every': checkpoint_every,
                'eval_every': eval_every,
                'hellaswag_every': hellaswag_every,
            },
            'final_results': {
                'final_step': step,
                'final_val_loss': final_val_loss,
                'final_val_accuracy': final_val_acc,
                'final_val_perplexity': final_val_perp,
                'total_training_time_minutes': total_time / 60,
            },
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'val_perplexities': val_perplexities,
                'step_times': step_times,
            },
            'checkpoints': checkpoint_info,
            'hellaswag_results': hellaswag_results,
        }
        
        # Save results
        results_file = self.output_dir / "exp3_extended_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Extended Training completed!")
        print(f"📊 Final Results:")
        print(f"   Final Step: {step}")
        print(f"   Final Val Loss: {final_val_loss:.6f}")
        print(f"   Final Val Accuracy: {final_val_acc:.6f}")
        print(f"   Final Val Perplexity: {final_val_perp:.4f}")
        print(f"   Total Training Time: {total_time/60:.2f} minutes")
        print(f"📁 Results saved to: {self.output_dir}")
        
        return results
    
    def create_training_curves(self, train_losses, val_losses, val_accuracies, val_perplexities):
        """Create training curves visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        ax1.plot(train_losses, 'b-', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Validation loss
        eval_steps = np.arange(0, len(train_losses), 100)  # Assuming eval every 100 steps
        ax2.plot(eval_steps[:len(val_losses)], val_losses, 'r-', linewidth=2, markersize=4)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Validation accuracy
        ax3.plot(eval_steps[:len(val_accuracies)], val_accuracies, 'g-', linewidth=2, markersize=4)
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Validation Accuracy')
        ax3.set_title('Validation Accuracy Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Validation perplexity
        ax4.plot(eval_steps[:len(val_perplexities)], val_perplexities, 'm-', linewidth=2, markersize=4)
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Validation Perplexity')
        ax4.set_title('Validation Perplexity Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "exp3_training_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curves saved to: {plot_file}")


def run_extended_training():
    """Run extended training for DeepSeek Attention + GLM4 MoE"""
    print("🚀 Extended Training for DeepSeek Attention + GLM4 MoE")
    print("=" * 60)
    
    # Create configuration
    config = MoEModelConfig(
        max_steps=10000,  # Will be overridden
        batch_size=16,    # Smaller batch for extended training
        max_tokens=50000, # Reduced tokens for memory
        eval_every=100,
        num_documents=1000,
        max_seq_len=128,  # Reduced sequence length
        d_model=256,     # Reduced model size
        n_heads=8,
        n_layers=6,      # Reduced layers
        d_ff=512,        # Smaller for MoE
        num_experts=4,   # Use reasonable default
        expert_top_k=2,  # Use reasonable default
    )
    
    # Create trainer
    trainer = ExtendedExperiment3Trainer(config, output_dir="exp3_results")
    
    # Run extended training
    results = trainer.run_extended_training(
        total_steps=10000,
        checkpoint_every=3000,
        eval_every=100,
        hellaswag_every=1000,
        learning_rate=3e-3  # Use optimal LR from search
    )
    
    return results


def main():
    """Main function to run extended training"""
    print("🚀 Extended Training for DeepSeek Attention + GLM4 MoE")
    print("=" * 60)
    
    try:
        results = run_extended_training()
        
        print(f"\n✅ Extended Training completed successfully!")
        print(f"📁 Results saved in: exp3_results/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in extended training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
