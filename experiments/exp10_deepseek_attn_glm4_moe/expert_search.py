#!/usr/bin/env python3
"""
Expert Configuration Search for Experiment 10: DeepSeek Attention + GLM4 MoE
Runs expert configuration search for optimal MoE setup
"""

import torch
import json
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
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
from experiments.exp8.exp8_models import AttentionMoE_8e_2k_512dModel


class ExpertConfigSearchTrainer:
    """Expert configuration search trainer for DeepSeek Attention + GLM4 MoE model"""
    
    def __init__(self, base_config: MoEModelConfig, output_dir: str = "expert_search_results"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data once
        print("📚 Loading data...")
        self.train_data, self.val_data = load_and_cache_data(base_config)
        
        print(f"✅ Data loaded: {len(self.train_data)} train, {len(self.val_data)} val samples")
    
    def train_with_expert_config(self, num_experts: int, expert_top_k: int, learning_rate: float = 1e-3, max_steps: int = 800) -> Dict[str, Any]:
        """Train model with specific expert configuration"""
        print(f"\n🧪 Training with {num_experts} experts, top-{expert_top_k}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create config for this expert setup
        config = MoEModelConfig(
            max_steps=max_steps,
            batch_size=self.base_config.batch_size,
            max_tokens=self.base_config.max_tokens,
            eval_every=100,
            num_documents=self.base_config.num_documents,
            max_seq_len=self.base_config.max_seq_len,
            d_model=self.base_config.d_model,
            n_heads=self.base_config.n_heads,
            n_layers=self.base_config.n_layers,
            d_ff=self.base_config.d_ff,
            num_experts=num_experts,
            expert_top_k=expert_top_k,
        )
        
        # Create model
        model = AttentionMoE_8e_2k_512dModel(config)
        model = model.cuda() if torch.cuda.is_available() else model
        
        # Create data loaders
        train_dataset = TextTokenDataset(self.train_data, config.max_seq_len)
        val_dataset = TextTokenDataset(self.val_data, config.max_seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Setup optimizer
        optimizer = setup_muon_optimizer(model, learning_rate=learning_rate)
        
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
                input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
                labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(input_ids)
                
                # Calculate loss
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    labels.view(-1), 
                    ignore_index=-100
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                step += 1
                
                # Evaluation
                if step % 100 == 0:
                    val_loss, val_acc, val_perp = self.evaluate(model, val_loader)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)
                    val_perplexities.append(val_perp)
                    
                    print(f"  Step {step}: Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_val_loss, final_val_acc, final_val_perp = self.evaluate(model, val_loader)
        
        return {
            'num_experts': num_experts,
            'expert_top_k': expert_top_k,
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
                input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
                labels = batch['labels'].cuda() if torch.cuda.is_available() else batch['labels']
                
                outputs = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)), 
                    labels.view(-1), 
                    ignore_index=-100
                )
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=-1)
                mask = labels != -100
                total_correct += ((predictions == labels) & mask).sum().item()
                total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        model.train()
        return avg_loss, accuracy, perplexity
    
    def run_expert_search(self, expert_configs: List[Tuple[int, int]], learning_rate: float = 1e-3, max_steps: int = 800) -> Dict[str, Any]:
        """Run expert configuration search"""
        results = {}
        
        for num_experts, expert_top_k in expert_configs:
            try:
                result = self.train_with_expert_config(num_experts, expert_top_k, learning_rate, max_steps)
                config_key = f"experts_{num_experts}_top_{expert_top_k}"
                results[config_key] = result
                
                # Save individual result
                result_file = self.output_dir / f"{config_key}_result.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
            except Exception as e:
                print(f"❌ Error training with {num_experts} experts, top-{expert_top_k}: {e}")
                config_key = f"experts_{num_experts}_top_{expert_top_k}"
                results[config_key] = {'error': str(e)}
        
        # Save all results
        results_file = self.output_dir / "expert_search_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create comparison plots
        self.create_comparison_plots(results)
        
        return results
    
    def create_comparison_plots(self, results: Dict[str, Any]):
        """Create comparison plots for expert search results"""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("❌ No valid results to plot")
            return
        
        # Extract data for plotting
        configs = []
        val_losses = []
        val_accuracies = []
        val_perplexities = []
        num_experts_list = []
        top_k_list = []
        
        for key, result in valid_results.items():
            configs.append(key)
            val_losses.append(result['val_loss'])
            val_accuracies.append(result['val_accuracy'])
            val_perplexities.append(result['val_perplexity'])
            num_experts_list.append(result['num_experts'])
            top_k_list.append(result['expert_top_k'])
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Validation Loss vs Expert Count
        ax1.plot(num_experts_list, val_losses, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Experts')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Validation Loss vs Number of Experts')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Validation Accuracy vs Expert Count
        ax2.plot(num_experts_list, val_accuracies, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Experts')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Validation Accuracy vs Number of Experts')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Validation Loss vs Top-k
        ax3.plot(top_k_list, val_losses, 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Top-k')
        ax3.set_ylabel('Validation Loss')
        ax3.set_title('Validation Loss vs Top-k')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Validation Accuracy vs Top-k
        ax4.plot(top_k_list, val_accuracies, 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Top-k')
        ax4.set_ylabel('Validation Accuracy')
        ax4.set_title('Validation Accuracy vs Top-k')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "expert_search_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plots saved to: {plot_file}")


def run_expert_search():
    """Run expert configuration search for DeepSeek Attention + GLM4 MoE model"""
    print("🚀 Expert Configuration Search for DeepSeek Attention + GLM4 MoE")
    print("=" * 60)
    
    # Create base configuration
    base_config = MoEModelConfig(
        max_steps=800,   # Steps per expert config
        batch_size=128,
        max_tokens=100000,
        eval_every=100,
        num_documents=1000,
        max_seq_len=256,
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=1024,  # Smaller for MoE
        num_experts=8,  # Default, will be overridden
        expert_top_k=2,  # Default, will be overridden
    )
    
    print(f"📋 Base Configuration:")
    print(f"   Model: DeepSeek Attention + GLM4 MoE 512d")
    print(f"   Batch Size: {base_config.batch_size}")
    print(f"   Model: {base_config.d_model}d, {base_config.n_layers}L, {base_config.n_heads}H")
    print(f"   Max Steps per Config: {base_config.max_steps}")
    
    # Define expert configurations to test
    expert_configs = [
        (4, 1),   # 4 experts, top-1
        (4, 2),   # 4 experts, top-2
        (8, 1),   # 8 experts, top-1
        (8, 2),   # 8 experts, top-2
        (16, 1),  # 16 experts, top-1
        (16, 2),  # 16 experts, top-2
    ]
    
    print(f"📋 Expert configurations to test:")
    for num_experts, top_k in expert_configs:
        print(f"   • {num_experts} experts, top-{top_k}")
    
    # Create expert search trainer
    expert_trainer = ExpertConfigSearchTrainer(base_config, output_dir="expert_search_results")
    
    # Run expert configuration search
    print(f"\n🧪 Running expert configuration search...")
    start_time = time.time()
    
    expert_results = expert_trainer.run_expert_search(
        expert_configs=expert_configs,
        learning_rate=1e-3,  # Use a reasonable LR
        max_steps=800
    )
    
    search_time = time.time() - start_time
    
    print(f"\n✅ Expert Configuration Search completed in {search_time/60:.2f} minutes!")
    
    # Find best expert configuration
    valid_results = {k: v for k, v in expert_results.items() if 'error' not in v}
    if valid_results:
        best_config_key = min(valid_results.keys(), key=lambda k: valid_results[k]['val_loss'])
        best_result = valid_results[best_config_key]
        
        print(f"\n🏆 Best Expert Configuration Found:")
        print(f"   Experts: {best_result['num_experts']}")
        print(f"   Top-k: {best_result['expert_top_k']}")
        print(f"   Validation Loss: {best_result['val_loss']:.6f}")
        print(f"   Validation Accuracy: {best_result['val_accuracy']:.6f}")
        print(f"   Validation Perplexity: {best_result['val_perplexity']:.4f}")
        print(f"   Training Steps: {best_result['total_steps']}")
        print(f"   Training Time: {best_result['training_time_minutes']:.2f} min")
        
        # Save best expert configuration recommendation
        recommendation = {
            'best_num_experts': best_result['num_experts'],
            'best_expert_top_k': best_result['expert_top_k'],
            'best_result': best_result,
            'all_results': expert_results,
            'search_time_minutes': search_time / 60,
            'recommendation': f"Use {best_result['num_experts']} experts with top-{best_result['expert_top_k']} for training"
        }
        
        recommendation_file = Path("expert_search_results") / "expert_recommendation.json"
        with open(recommendation_file, 'w') as f:
            json.dump(recommendation, f, indent=2)
        
        print(f"\n💾 Recommendation saved to: {recommendation_file}")
        
        return best_result['num_experts'], best_result['expert_top_k'], expert_results
    else:
        print("❌ No valid expert configuration results found!")
        return None, None, expert_results


def main():
    """Main function to run expert configuration search"""
    print("🚀 Expert Configuration Search for DeepSeek Attention + GLM4 MoE")
    print("=" * 60)
    
    try:
        best_num_experts, best_top_k, expert_results = run_expert_search()
        
        if best_num_experts is None:
            print("❌ Expert configuration search failed!")
            return False
        
        print(f"\n✅ Expert Configuration Search completed successfully!")
        print(f"📁 Results saved in: expert_search_results/")
        print(f"🎯 Recommended Config: {best_num_experts} experts, top-{best_top_k}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in expert configuration search: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
