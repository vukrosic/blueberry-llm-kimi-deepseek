#!/usr/bin/env python3
"""
Learning Rate Search for Experiment 10: DeepSeek Attention + GLM4 MoE
Runs learning rate search for DeepSeek Attention + GLM4 MoE model
"""

import torch
import json
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
from torch.utils.data import DataLoader

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


class LearningRateSearchTrainer:
    """Learning rate search trainer for DeepSeek Attention + GLM4 MoE model"""
    
    def __init__(self, config: MoEModelConfig, output_dir: str = "lr_search_results"):
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
    
    def train_with_lr(self, learning_rate: float, max_steps: int = 1000, eval_every: int = 100) -> Dict[str, Any]:
        """Train model with specific learning rate"""
        print(f"\n🧪 Training with LR={learning_rate:.2e}")
        
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
        
        # Training loop
        model.train()
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_perplexities = []
        
        step = 0
        start_time = time.time()
        
        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break
                    
                # Move batch to device
                x, y = batch
                input_ids = x.cuda() if torch.cuda.is_available() else x
                labels = y.cuda() if torch.cuda.is_available() else y
                
                # Forward pass
                for optimizer in optimizers:
                    optimizer.zero_grad()
                outputs, aux_loss = model(input_ids, return_aux_loss=True)
                
                # Calculate loss
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    labels.view(-1)
                )
                
                # Add auxiliary loss if present
                if aux_loss is not None:
                    loss = loss + aux_loss
                
                # Backward pass
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                
                train_losses.append(loss.item())
                step += 1
                
                # Evaluation
                if step % eval_every == 0:
                    val_loss, val_acc, val_perp = self.evaluate(model, val_loader)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)
                    val_perplexities.append(val_perp)
                    
                    print(f"  Step {step}: Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_val_loss, final_val_acc, final_val_perp = self.evaluate(model, val_loader)
        
        return {
            'learning_rate': learning_rate,
            'val_loss': final_val_loss,
            'val_accuracy': final_val_acc,
            'val_perplexity': final_val_perp,
            'total_steps': step,
            'training_time_minutes': training_time / 60,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_perplexities': val_perplexities
        }
    
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
    
    def run_lr_search(self, learning_rates: List[float], max_steps: int = 1000, eval_every: int = 100) -> Dict[str, Any]:
        """Run learning rate search"""
        results = {}
        
        for lr in learning_rates:
            try:
                result = self.train_with_lr(lr, max_steps, eval_every)
                results[f"lr_{lr:.2e}"] = result
                
                # Save individual result
                result_file = self.output_dir / f"lr_{lr:.2e}_result.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
            except Exception as e:
                print(f"❌ Error training with LR={lr:.2e}: {e}")
                results[f"lr_{lr:.2e}"] = {'error': str(e)}
        
        # Save all results
        results_file = self.output_dir / "lr_search_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create comparison plot
        self.create_comparison_plot(results)
        
        return results
    
    def create_comparison_plot(self, results: Dict[str, Any]):
        """Create comparison plot for learning rate search results"""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("❌ No valid results to plot")
            return
        
        lrs = []
        val_losses = []
        val_accuracies = []
        
        for key, result in valid_results.items():
            lrs.append(result['learning_rate'])
            val_losses.append(result['val_loss'])
            val_accuracies.append(result['val_accuracy'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Validation loss plot
        ax1.semilogx(lrs, val_losses, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Learning Rate vs Validation Loss')
        ax1.grid(True, alpha=0.3)
        
        # Validation accuracy plot
        ax2.semilogx(lrs, val_accuracies, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Learning Rate vs Validation Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "lr_search_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plot saved to: {plot_file}")


def run_lr_search():
    """Run learning rate search for DeepSeek Attention + GLM4 MoE model"""
    print("🚀 Learning Rate Search for DeepSeek Attention + GLM4 MoE")
    print("=" * 60)
    
    # Create configuration
    config = MoEModelConfig(
        max_steps=1000,   # Steps per learning rate
        batch_size=32,    # Reduced batch size for memory
        max_tokens=50000, # Reduced tokens for memory
        eval_every=100,   # Evaluate every 100 steps
        num_documents=1000,
        max_seq_len=128,  # Reduced sequence length
        d_model=256,     # Reduced model size
        n_heads=8,
        n_layers=6,      # Reduced layers
        d_ff=512,        # Smaller for MoE
        num_experts=4,   # Reduced experts
        expert_top_k=2,
    )
    
    print(f"📋 Configuration:")
    print(f"   Model: DeepSeek Attention + GLM4 MoE {config.d_model}d")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Model: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"   MoE: {config.num_experts} experts, top-{config.expert_top_k}")
    print(f"   Max Steps per LR: {config.max_steps}")
    
    # Define learning rates to test
    learning_rates = [
        1e-4,   # Low
        3e-4,   # Medium-low
        1e-3,   # Medium-high
        3e-3,   # High
    ]
    
    print(f"📋 Learning rates to test: {learning_rates}")
    
    # Create LR search trainer
    lr_trainer = LearningRateSearchTrainer(config, output_dir="lr_search_results")
    
    # Run learning rate search
    print(f"\n🧪 Running learning rate search...")
    start_time = time.time()
    
    lr_results = lr_trainer.run_lr_search(
        learning_rates=learning_rates,
        max_steps=1000,
        eval_every=100
    )
    
    search_time = time.time() - start_time
    
    print(f"\n✅ Learning Rate Search completed in {search_time/60:.2f} minutes!")
    
    # Find best learning rate
    valid_results = {k: v for k, v in lr_results.items() if 'error' not in v}
    if valid_results:
        best_lr_key = min(valid_results.keys(), key=lambda k: valid_results[k]['val_loss'])
        best_result = valid_results[best_lr_key]
        best_lr = best_result['learning_rate']
        
        print(f"\n🏆 Best Learning Rate Found:")
        print(f"   Learning Rate: {best_lr:.2e}")
        print(f"   Validation Loss: {best_result['val_loss']:.6f}")
        print(f"   Validation Accuracy: {best_result['val_accuracy']:.6f}")
        print(f"   Validation Perplexity: {best_result['val_perplexity']:.4f}")
        print(f"   Training Steps: {best_result['total_steps']}")
        print(f"   Training Time: {best_result['training_time_minutes']:.2f} min")
        
        # Save best LR recommendation
        recommendation = {
            'best_learning_rate': best_lr,
            'best_result': best_result,
            'all_results': lr_results,
            'search_time_minutes': search_time / 60,
            'recommendation': f"Use learning rate {best_lr:.2e} for training"
        }
        
        recommendation_file = Path("lr_search_results") / "lr_recommendation.json"
        with open(recommendation_file, 'w') as f:
            json.dump(recommendation, f, indent=2)
        
        print(f"\n💾 Recommendation saved to: {recommendation_file}")
        
        return best_lr, lr_results
    else:
        print("❌ No valid learning rate results found!")
        return None, lr_results


def main():
    """Main function to run learning rate search"""
    print("🚀 Learning Rate Search for DeepSeek Attention + GLM4 MoE")
    print("=" * 60)
    
    try:
        best_lr, lr_results = run_lr_search()
        
        if best_lr is None:
            print("❌ Learning rate search failed!")
            return False
        
        print(f"\n✅ Learning Rate Search completed successfully!")
        print(f"📁 Results saved in: lr_search_results/")
        print(f"🎯 Recommended LR: {best_lr:.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in learning rate search: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
