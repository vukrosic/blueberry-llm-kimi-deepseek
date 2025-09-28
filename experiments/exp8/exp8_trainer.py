"""
Experiment 8: Reduced Ablation Study Trainer
Focused on 512 hidden dimension scale with powers of 2 ablations
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
from experiments.exp8.exp8_reduced_ablation_models import (
    create_reduced_ablation_model, 
    REDUCED_ABLATION_MODELS,
    print_reduced_ablation_summary
)
from benchmark_evaluator import HellaSwagEvaluator


class ReducedExperiment8Trainer:
    """Reduced ablation study trainer focused on 512 scale"""
    
    def __init__(self, base_config: MoEModelConfig, output_dir: str = "experiments/exp8/exp8_results_600steps"):
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
        
        # Initialize HellaSwag evaluator
        self.hellaswag_evaluator = HellaSwagEvaluator(output_dir=str(self.output_dir / "hellaswag_benchmark"))
    
    def train_model(self, model, model_name: str, test_steps: int = 100) -> Dict[str, Any]:
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
                
                # Progress logging (every 50 steps)
                if step % 50 == 0:
                    print(f"Step {step}/{test_steps}: Loss={ce_loss.item():.4f}")
                
                # Evaluation (every 50 steps)
                if step % 50 == 0 and step > 0:
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
    
    def run_reduced_test(self, test_steps: int = 100) -> Dict[str, Any]:
        """Run reduced ablation test on all 13 models"""
        print(f"\n🚀 Starting Reduced Ablation Experiment 8: 512 Scale Focus")
        print(f"📋 Testing {len(REDUCED_ABLATION_MODELS)} models with {test_steps} steps each")
        
        # Print summary of all models
        print_reduced_ablation_summary()
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Test all models
        results = {}
        successful = 0
        failed = 0
        
        for i, model_name in enumerate(REDUCED_ABLATION_MODELS.keys(), 1):
            try:
                print(f"\n🧪 [{i}/{len(REDUCED_ABLATION_MODELS)}] Testing {model_name}...")
                
                # Clear GPU memory before each experiment
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create model
                model = create_reduced_ablation_model(model_name, self.base_config)
                
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
        self._save_results(results, "reduced")
        
        # Print reduced ablation comparison
        self._print_reduced_comparison(results)
        
        # Create reduced ablation loss vs time plot
        self._create_reduced_plot()
        
        # Run HellaSwag benchmark evaluation on all models
        print(f"\n🧪 Running HellaSwag benchmark evaluation...")
        benchmark_results = self.hellaswag_evaluator.evaluate_all_models(results)
        
        # Add benchmark results to main results
        for model_name, benchmark_result in benchmark_results.items():
            if model_name in results:
                results[model_name]['hellaswag_benchmark'] = benchmark_result
        
        print(f"\n📊 Reduced Ablation Test Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📁 Results saved in: {self.output_dir}")
        print(f"   🧪 HellaSwag benchmark evaluated: {len(benchmark_results)} models")
        
        return results
    
    def _save_results(self, results: Dict[str, Any], mode: str):
        """Save experiment results to file"""
        results_file = self.output_dir / f"exp8_{mode}_results_600steps.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def _print_reduced_comparison(self, results: Dict[str, Any]):
        """Print reduced ablation comparison of all 13 models"""
        print(f"\n{'='*120}")
        print(f"📊 REDUCED ABLATION EXPERIMENT 8 COMPARISON: All {len(REDUCED_ABLATION_MODELS)} Models")
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
        
        print(f"{'='*120}")
    
    def _get_model_category(self, model_name: str) -> str:
        """Get category for model"""
        if model_name == "baseline":
            return "Baseline"
        elif model_name.startswith("mlp_") and "attention" not in model_name:
            return "MLP Scaling"
        elif model_name.startswith("attention_mlp_"):
            return "Attention+MLP"
        elif model_name.startswith("moe_") and "attention" not in model_name:
            return "MoE Scaling"
        elif model_name.startswith("attention_moe_"):
            return "Attention+MoE"
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
    
    def _create_reduced_plot(self):
        """Create reduced ablation loss vs time visualization"""
        if not self.loss_curves:
            return
        
        # Color scheme by category
        colors = {
            'Baseline': 'black',
            'MLP Scaling': 'blue',
            'Attention+MLP': 'green', 
            'MoE Scaling': 'orange',
            'Attention+MoE': 'red'
        }
        
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
        
        # Prepare data for plotting
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
                
                plot_data.append({
                    'name': name,
                    'times': elapsed_times,
                    'losses': data['eval_losses'],
                    'category': category,
                    'color': color,
                    'marker': marker,
                    'final_loss': data['eval_losses'][-1]
                })
                all_final_losses.append(data['eval_losses'][-1])
        
        # Create main plot
        plt.figure(figsize=(16, 10))
        
        for data in plot_data:
            plt.plot(data['times'], data['losses'], 
                    color=data['color'], marker=data['marker'], linewidth=2, markersize=6,
                    label=f'{data["name"]} (Final: {data["final_loss"]:.4f})', alpha=0.8)
        
        plt.xlabel('Time (minutes)', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title(f'Reduced Ablation Experiment 8: Validation Loss vs Time (512 Scale Focus)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "exp8_reduced_loss_vs_time_comparison_600steps.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Loss visualization created: {plot_file}")


def main():
    """Main function to run Reduced Ablation Experiment 8"""
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create configuration for reduced ablation testing
    # Based on the provided JSON config, targeting 512 scale
    base_config = MoEModelConfig(
        max_steps=600,  # 600 steps per model for extended testing
        batch_size=16,
        max_tokens=100000,
        eval_every=50,  # Evaluation every 50 steps
        num_documents=1000,
        max_seq_len=256,
        d_model=512,  # Target 512 scale
        n_heads=8,    # Powers of 2
        n_layers=3,   # Keep same layers
        d_ff=2048,    # Powers of 2 (4x d_model)
        num_experts=8,  # Powers of 2
        expert_top_k=2,  # Keep same top-k
    )
    
    print(f"🚀 Reduced Ablation Experiment 8 Configuration:")
    print(f"   Steps: {base_config.max_steps} (600 steps per model)")
    print(f"   Batch Size: {base_config.batch_size}")
    print(f"   Model: {base_config.d_model}d, {base_config.n_layers}L, {base_config.n_heads}H")
    print(f"   MoE: {base_config.num_experts} experts, top-{base_config.expert_top_k}")
    print(f"   Models: {len(REDUCED_ABLATION_MODELS)} reduced configurations")
    print(f"   Expected Total Time: ~{len(REDUCED_ABLATION_MODELS) * 60} minutes (600 steps each)")
    
    # Create trainer
    trainer = ReducedExperiment8Trainer(base_config)
    
    # Run reduced ablation study
    print(f"\n🧪 Running reduced ablation study (600 steps per model)...")
    
    # Run reduced ablation experiment
    results = trainer.run_reduced_test(test_steps=600)
    
    print(f"\n✅ Reduced Ablation Experiment 8 completed!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
