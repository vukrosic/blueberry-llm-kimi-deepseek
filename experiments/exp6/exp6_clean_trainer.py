"""
Experiment 6: Clean Ablation Study Trainer
Testing meaningful combinations without RMSNorm and with clear MoE naming
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
from experiments.exp6.exp6_clean_ablation_models import (
    create_clean_ablation_model, 
    CLEAN_ABLATION_MODELS,
    print_clean_ablation_summary
)


class CleanExperiment6Trainer:
    """Clean ablation study trainer for meaningful model configurations"""
    
    def __init__(self, base_config: MoEModelConfig, output_dir: str = "experiments/exp6/exp6_clean_results_extended"):
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
    
    def train_model(self, model, model_name: str, test_steps: int = 20) -> Dict[str, Any]:
        """Train a single model with comprehensive metrics"""
        print(f"\n{'='*80}")
        print(f"🧪 Training: {model_name} ({test_steps} steps)")
        print(f"{'='*80}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizer
        optimizers = setup_muon_optimizer(model, self.base_config)
        
        # Learning rate schedule
        schedulers = []
        for optimizer in optimizers:
            warmup_steps = max(1, test_steps // 10)  # 10% warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (test_steps - warmup_steps)
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
        
        print(f"🚀 Starting training for {test_steps} steps...")
        
        while step < test_steps:
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= test_steps:
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
                
                # Progress logging (every 25 steps for longer training)
                if step % 25 == 0:  # Log every 25 steps
                    print(f"Step {step}/{test_steps}: Loss={ce_loss.item():.4f}")
                
                # Evaluation (every 25 steps for longer training)
                if step % 25 == 0 and step > 0:
                    eval_metrics = self._evaluate_model(model, val_loader)
                    
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
        eval_steps.append(test_steps)
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
            'parameters_millions': param_count / 1e6,
            'test_steps': test_steps
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
        
        # Limit evaluation to first 3 batches for speed
        max_eval_batches = 3
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
    
    def run_clean_test(self, test_steps: int = 100) -> Dict[str, Any]:
        """Run size ablation test on all 32 models"""
        print(f"\n🚀 Starting Size Ablation Experiment 6: MLP vs MoE Size Comparison")
        print(f"📋 Testing {len(CLEAN_ABLATION_MODELS)} models with {test_steps} steps each")
        
        # Print summary of all models
        print_clean_ablation_summary()
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Test all models
        results = {}
        successful = 0
        failed = 0
        
        for i, model_name in enumerate(CLEAN_ABLATION_MODELS.keys(), 1):
            try:
                print(f"\n🧪 [{i}/{len(CLEAN_ABLATION_MODELS)}] Testing {model_name}...")
                
                # Clear GPU memory before each experiment
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create model
                model = create_clean_ablation_model(model_name, self.base_config)
                
                # Train for test_steps
                result = self.train_model(model, model_name, test_steps=test_steps)
                results[model_name] = result
                successful += 1
                
                print(f"✅ {model_name} test completed successfully")
                
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
                results[model_name] = {"error": str(e), "model_name": model_name}
                failed += 1
        
        # Store results
        self.results = results
        
        # Save results
        self._save_results(results, "clean")
        
        # Print size ablation comparison
        self._print_clean_comparison(results)
        
        # Create size ablation loss vs time plot
        self._create_clean_plot()
        
        print(f"\n📊 Size Ablation Test Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📁 Results saved in: {self.output_dir}")
        
        return results
    
    def _save_results(self, results: Dict[str, Any], mode: str):
        """Save experiment results to file"""
        results_file = self.output_dir / f"exp6_clean_{mode}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def _print_clean_comparison(self, results: Dict[str, Any]):
        """Print size ablation comparison of all 32 models"""
        print(f"\n{'='*120}")
        print(f"📊 SIZE ABLATION EXPERIMENT 6 COMPARISON: All {len(CLEAN_ABLATION_MODELS)} Models")
        print(f"{'='*120}")
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not successful_results:
            print("❌ No successful experiments")
            return
        
        # Sort by validation loss
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]["val_loss"])
        
        print(f"{'Rank':<4} {'Model':<35} {'Val Loss':<10} {'Val Acc':<10} {'Val Perp':<10} {'Time (min)':<12} {'Params (M)':<12} {'Category':<20}")
        print(f"{'-'*120}")
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            val_loss = result.get('val_loss', 0)
            val_acc = result.get('val_accuracy', 0)
            val_perp = result.get('val_perplexity', 0)
            time_min = result.get('training_time_minutes', 0)
            params_m = result.get('parameters_millions', 0)
            
            # Get category
            category = self._get_model_category(name)
            
            print(f"{rank:<4} {name:<35} {val_loss:<10.4f} {val_acc:<10.4f} {val_perp:<10.2f} {time_min:<12.2f} {params_m:<12.2f} {category:<20}")
        
        print(f"{'-'*120}")
        
        # Find best model
        if len(sorted_results) > 0:
            best_model = sorted_results[0]
            print(f"\n🏆 Best Model: {best_model[0]} (Loss: {best_model[1]['val_loss']:.4f})")
        
        # Category analysis
        self._print_category_analysis(successful_results)
        
        # Statistical analysis
        self._print_statistical_analysis(successful_results)
        
        print(f"{'='*120}")
    
    def _get_model_category(self, model_name: str) -> str:
        """Get category for model"""
        if model_name == "baseline":
            return "Baseline"
        elif model_name in ["mlp", "attention"]:
            return "Single Component"
        elif "moe_" in model_name and "attention" not in model_name:
            return "MoE Only"
        elif "attention_moe_" in model_name:
            if "_256d" in model_name or "_512d" in model_name or "_1024d" in model_name:
                return "Architecture Scaling"
            elif "_3layers" in model_name or "_6layers" in model_name:
                return "Layer Count"
            elif "_no_rope" in model_name or "_no_bias" in model_name or "standard_" in model_name:
                return "Attention Variant"
            else:
                return "Two Components"
        elif model_name == "attention_mlp":
            return "Two Components"
        else:
            return "Other"
    
    def _print_category_analysis(self, results: Dict[str, Any]):
        """Print analysis by category"""
        print(f"\n🔬 Category Analysis:")
        
        categories = {}
        for name, result in results.items():
            category = self._get_model_category(name)
            if category not in categories:
                categories[category] = []
            categories[category].append((name, result))
        
        for category, models in categories.items():
            if not models:
                continue
                
            losses = [result['val_loss'] for _, result in models]
            best_loss = min(losses)
            avg_loss = np.mean(losses)
            
            print(f"\n   📋 {category} ({len(models)} models):")
            print(f"      Best Loss: {best_loss:.4f}")
            print(f"      Avg Loss: {avg_loss:.4f}")
            
            # Show best model in category
            best_model = min(models, key=lambda x: x[1]['val_loss'])
            print(f"      Best: {best_model[0]} ({best_model[1]['val_loss']:.4f})")
    
    def _print_statistical_analysis(self, results: Dict[str, Any]):
        """Print statistical analysis of results"""
        if len(results) < 2:
            return
        
        print(f"\n📈 Statistical Analysis:")
        
        # Extract metrics
        losses = [result['val_loss'] for result in results.values()]
        accuracies = [result['val_accuracy'] for result in results.values()]
        times = [result['training_time_minutes'] for result in results.values()]
        params = [result['parameters_millions'] for result in results.values()]
        
        print(f"   Loss Statistics:")
        print(f"     Mean: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"     Range: {np.min(losses):.4f} - {np.max(losses):.4f}")
        print(f"     Best/Worst Ratio: {np.min(losses)/np.max(losses):.2f}x")
        
        print(f"   Accuracy Statistics:")
        print(f"     Mean: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"     Range: {np.min(accuracies):.4f} - {np.max(accuracies):.4f}")
        
        print(f"   Time Statistics:")
        print(f"     Mean: {np.mean(times):.1f} ± {np.std(times):.1f} min")
        print(f"     Range: {np.min(times):.1f} - {np.max(times):.1f} min")
        
        print(f"   Parameter Statistics:")
        print(f"     Mean: {np.mean(params):.1f} ± {np.std(params):.1f}M")
        print(f"     Range: {np.min(params):.1f} - {np.max(params):.1f}M")
    
    def _create_clean_plot(self):
        """Create multiple clean loss vs time visualizations with better scaling"""
        if not self.loss_curves:
            return
        
        # Color scheme by category
        colors = {
            'Baseline': 'black',
            'Single Component': 'blue',
            'MoE Only': 'green', 
            'Two Components': 'orange',
            'Architecture Scaling': 'red',
            'Layer Count': 'purple',
            'Attention Variant': 'brown'
        }
        
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
        
        # Prepare data for all plots
        plot_data = []
        all_final_losses = []
        
        for i, (name, data) in enumerate(self.loss_curves.items()):
            if 'eval_times' in data and len(data['eval_times']) > 1:
                # Convert timestamps to elapsed time in minutes
                start_time = data['eval_times'][0]
                elapsed_times = [(t - start_time) / 60 for t in data['eval_times']]
                
                category = self._get_model_category(name)
                color = colors.get(category, 'gray')
                marker = markers[i % len(markers)]
                
                # Calculate normalized loss (relative to initial loss)
                initial_loss = data['eval_losses'][0]
                normalized_losses = [loss / initial_loss for loss in data['eval_losses']]
                
                plot_data.append({
                    'name': name,
                    'times': elapsed_times,
                    'losses': data['eval_losses'],
                    'normalized_losses': normalized_losses,
                    'category': category,
                    'color': color,
                    'marker': marker,
                    'final_loss': data['eval_losses'][-1],
                    'initial_loss': initial_loss
                })
                all_final_losses.append(data['eval_losses'][-1])
        
        # 1. Original plot with better y-axis limits
        plt.figure(figsize=(20, 12))
        
        for data in plot_data:
            plt.plot(data['times'], data['losses'], 
                    color=data['color'], marker=data['marker'], linewidth=2, markersize=6,
                    label=f'{data["name"]} (Final: {data["final_loss"]:.4f})', alpha=0.8)
        
        # Set y-axis limits to focus on the meaningful range
        min_final = min(all_final_losses)
        max_initial = max([d['initial_loss'] for d in plot_data])
        plt.ylim(min_final * 0.8, max_initial * 1.1)
        
        plt.xlabel('Time (minutes)', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title(f'Size Ablation Experiment 6: Validation Loss vs Time (Improved Scaling)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save improved plot
        plot_file = self.output_dir / "exp6_clean_loss_vs_time_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Normalized loss plot (relative improvement)
        plt.figure(figsize=(20, 12))
        
        for data in plot_data:
            plt.plot(data['times'], data['normalized_losses'], 
                    color=data['color'], marker=data['marker'], linewidth=2, markersize=6,
                    label=f'{data["name"]} (Improvement: {(1-data["normalized_losses"][-1])*100:.1f}%)', alpha=0.8)
        
        plt.xlabel('Time (minutes)', fontsize=12)
        plt.ylabel('Normalized Loss (Relative to Initial)', fontsize=12)
        plt.title(f'Size Ablation Experiment 6: Relative Loss Improvement vs Time', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Loss (100%)')
        plt.tight_layout()
        
        # Save normalized plot
        normalized_plot_file = self.output_dir / "exp6_clean_normalized_loss_comparison.png"
        plt.savefig(normalized_plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Zoomed plot focusing on final loss range
        plt.figure(figsize=(20, 12))
        
        for data in plot_data:
            plt.plot(data['times'], data['losses'], 
                    color=data['color'], marker=data['marker'], linewidth=2, markersize=6,
                    label=f'{data["name"]} (Final: {data["final_loss"]:.4f})', alpha=0.8)
        
        # Focus on final loss range with some margin
        final_losses = [d['final_loss'] for d in plot_data]
        min_final = min(final_losses)
        max_final = max(final_losses)
        margin = (max_final - min_final) * 0.2
        plt.ylim(min_final - margin, max_final + margin)
        
        plt.xlabel('Time (minutes)', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title(f'Size Ablation Experiment 6: Final Loss Range Focus (Zoomed View)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save zoomed plot
        zoomed_plot_file = self.output_dir / "exp6_clean_zoomed_loss_comparison.png"
        plt.savefig(zoomed_plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Category-based subplot
        categories = list(set([d['category'] for d in plot_data]))
        n_categories = len(categories)
        cols = 3
        rows = (n_categories + cols - 1) // cols
        
        plt.figure(figsize=(20, 5 * rows))
        
        for i, category in enumerate(categories):
            plt.subplot(rows, cols, i + 1)
            category_data = [d for d in plot_data if d['category'] == category]
            
            for data in category_data:
                plt.plot(data['times'], data['losses'], 
                        color=data['color'], marker=data['marker'], linewidth=2, markersize=6,
                        label=f'{data["name"]} (Final: {data["final_loss"]:.4f})', alpha=0.8)
            
            plt.xlabel('Time (minutes)', fontsize=10)
            plt.ylabel('Validation Loss', fontsize=10)
            plt.title(f'{category} Models', fontsize=12)
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save category plot
        category_plot_file = self.output_dir / "exp6_clean_category_loss_comparison.png"
        plt.savefig(category_plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Multiple loss visualizations created:")
        print(f"   📊 Main plot (improved scaling): {plot_file}")
        print(f"   📈 Normalized improvement plot: {normalized_plot_file}")
        print(f"   🔍 Zoomed final range plot: {zoomed_plot_file}")
        print(f"   📋 Category comparison plot: {category_plot_file}")


def main():
    """Main function to run Size Ablation Experiment 6"""
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create configuration for extended size ablation testing (500 steps each)
    base_config = MoEModelConfig(
        max_steps=500,  # 500 steps per model for better convergence
        batch_size=16,  # Same batch size as previous experiments
        max_tokens=100000,  # Same dataset size
        eval_every=25,  # Evaluation every 25 steps for longer training
        num_documents=1000,  # Same dataset size
        max_seq_len=256,  # Same sequence length
        d_model=256,  # Same model size
        n_heads=4,    # Same attention heads
        n_layers=3,   # Same number of layers
        d_ff=1024,    # Same feed-forward size
        num_experts=8,  # Same number of experts
        expert_top_k=2,  # Same top-k selection
    )
    
    print(f"🚀 Extended Size Ablation Experiment 6 Configuration:")
    print(f"   Steps: {base_config.max_steps} (500 steps per model for better curves)")
    print(f"   Batch Size: {base_config.batch_size}")
    print(f"   Model: {base_config.d_model}d, {base_config.n_layers}L, {base_config.n_heads}H")
    print(f"   MoE: {base_config.num_experts} experts, top-{base_config.expert_top_k}")
    print(f"   Models: {len(CLEAN_ABLATION_MODELS)} size ablation configurations")
    print(f"   Expected Total Time: ~{len(CLEAN_ABLATION_MODELS) * 50} minutes (500 steps each)")
    
    # Create trainer
    trainer = CleanExperiment6Trainer(base_config)
    
    # Run extended size ablation study with 500 steps per model
    print(f"\n🧪 Running extended size ablation study (500 steps per model)...")
    
    # Run extended size ablation experiment
    results = trainer.run_clean_test(test_steps=500)
    
    print(f"\n✅ Size Ablation Experiment 6 completed!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
