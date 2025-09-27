"""
Experiment 6: Comprehensive Ablation Study Trainer
Combining best components from experiments 1-5 with fair comparison
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
from experiments.exp6.exp6_ablation_models import create_ablation_model, ABLATION_MODELS


class Experiment6Trainer:
    """Comprehensive ablation study trainer"""
    
    def __init__(self, base_config: MoEModelConfig, output_dir: str = "experiments/exp6/exp6_results"):
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
    
    def train_model(self, model, model_name: str, test_mode: bool = False) -> Dict[str, Any]:
        """Train a single model with comprehensive metrics"""
        print(f"\n{'='*80}")
        print(f"🧪 Training: {model_name} {'(TEST MODE - 5 steps)' if test_mode else ''}")
        print(f"{'='*80}")
        
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
        
        max_steps = 5 if test_mode else self.base_config.max_steps
        print(f"🚀 Starting training for {max_steps} steps...")
        
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
                
                # Progress logging
                if step % 1 == 0:  # Log every step in test mode
                    print(f"Step {step}/{max_steps}: Loss={ce_loss.item():.4f}")
                
                # Evaluation (every step in test mode, every eval_every in full mode)
                eval_frequency = 1 if test_mode else self.base_config.eval_every
                if step % eval_frequency == 0 and step > 0:
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
        eval_steps.append(max_steps)
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
            'test_mode': test_mode,
            'max_steps': max_steps
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
    
    def run_test_experiment(self) -> Dict[str, Any]:
        """Run 5-step test to verify all models work"""
        print(f"\n🚀 Starting Experiment 6: Ablation Study TEST (5 steps each)")
        print(f"📋 Testing {len(ABLATION_MODELS)} models to verify they work correctly")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Test all models
        results = {}
        
        for model_name in ABLATION_MODELS.keys():
            try:
                print(f"\n🧪 Testing {model_name}...")
                
                # Clear GPU memory before each experiment
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create model
                model = create_ablation_model(model_name, self.base_config)
                
                # Train for 5 steps
                result = self.train_model(model, model_name, test_mode=True)
                results[model_name] = result
                
                print(f"✅ {model_name} test completed successfully")
                
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
                results[model_name] = {"error": str(e), "model_name": model_name}
        
        # Save test results
        self._save_results(results, "test")
        
        # Print test summary
        self._print_test_summary(results)
        
        return results
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run full ablation study experiment"""
        print(f"\n🚀 Starting Experiment 6: Comprehensive Ablation Study")
        print(f"📋 Comparing {len(ABLATION_MODELS)} model configurations")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Train all models
        results = {}
        
        for model_name in ABLATION_MODELS.keys():
            try:
                print(f"\n🧪 Training {model_name}...")
                
                # Clear GPU memory before each experiment
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create model
                model = create_ablation_model(model_name, self.base_config)
                
                # Train model
                result = self.train_model(model, model_name, test_mode=False)
                results[model_name] = result
                
                print(f"✅ {model_name} training completed")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                results[model_name] = {"error": str(e), "model_name": model_name}
        
        # Store results
        self.results = results
        
        # Save results
        self._save_results(results, "full")
        
        # Print comparison
        self._print_comparison(results)
        
        # Create loss vs time plot
        self._create_loss_vs_time_plot()
        
        return results
    
    def _save_results(self, results: Dict[str, Any], mode: str):
        """Save experiment results to file"""
        results_file = self.output_dir / f"exp6_ablation_{mode}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """Print test results summary"""
        print(f"\n{'='*80}")
        print(f"📊 EXPERIMENT 6 TEST SUMMARY: Ablation Study (5 steps each)")
        print(f"{'='*80}")
        
        successful = 0
        failed = 0
        
        for name, result in results.items():
            if "error" in result:
                print(f"❌ {name}: FAILED - {result['error']}")
                failed += 1
            else:
                print(f"✅ {name}: SUCCESS - Loss={result['val_loss']:.4f}, Time={result['training_time_minutes']:.2f}min")
                successful += 1
        
        print(f"\n📈 Test Results: {successful} successful, {failed} failed")
        print(f"{'='*80}")
    
    def _print_comparison(self, results: Dict[str, Any]):
        """Print comprehensive comparison of all models"""
        print(f"\n{'='*100}")
        print(f"📊 EXPERIMENT 6 COMPARISON: Comprehensive Ablation Study")
        print(f"{'='*100}")
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not successful_results:
            print("❌ No successful experiments")
            return
        
        # Sort by validation loss
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]["val_loss"])
        
        print(f"{'Model':<20} {'Val Loss':<10} {'Val Acc':<10} {'Val Perp':<10} {'Time (min)':<12} {'Params (M)':<12} {'Components':<30}")
        print(f"{'-'*120}")
        
        for name, result in sorted_results:
            val_loss = result.get('val_loss', 0)
            val_acc = result.get('val_accuracy', 0)
            val_perp = result.get('val_perplexity', 0)
            time_min = result.get('training_time_minutes', 0)
            params_m = result.get('parameters_millions', 0)
            
            # Get component description
            components = self._get_component_description(name)
            
            print(f"{name:<20} {val_loss:<10.4f} {val_acc:<10.4f} {val_perp:<10.2f} {time_min:<12.2f} {params_m:<12.2f} {components:<30}")
        
        print(f"{'-'*120}")
        
        # Find best model
        if len(sorted_results) > 0:
            best_model = sorted_results[0]
            print(f"\n🏆 Best Model: {best_model[0]} (Loss: {best_model[1]['val_loss']:.4f})")
        
        # Statistical analysis
        self._print_statistical_analysis(successful_results)
        
        print(f"{'='*120}")
    
    def _get_component_description(self, model_name: str) -> str:
        """Get human-readable description of model components"""
        descriptions = {
            "baseline": "None (control)",
            "rmsnorm": "DeepSeek RMSNorm",
            "mlp": "DeepSeek MLP",
            "moe": "GLM4 MoE",
            "attention": "DeepSeek Attention",
            "rmsnorm_mlp": "RMSNorm + MLP",
            "rmsnorm_moe": "RMSNorm + GLM4 MoE",
            "mlp_moe": "GLM4 MoE (replaces MLP)",
            "attention_rmsnorm": "Attention + RMSNorm",
            "attention_mlp": "Attention + MLP",
            "attention_moe": "Attention + GLM4 MoE",
            "all_components": "DeepSeek components + GLM4 MoE"
        }
        return descriptions.get(model_name, "Unknown")
    
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
        
        print(f"   Accuracy Statistics:")
        print(f"     Mean: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"     Range: {np.min(accuracies):.4f} - {np.max(accuracies):.4f}")
        
        print(f"   Time Statistics:")
        print(f"     Mean: {np.mean(times):.1f} ± {np.std(times):.1f} min")
        print(f"     Range: {np.min(times):.1f} - {np.max(times):.1f} min")
        
        print(f"   Parameter Statistics:")
        print(f"     Mean: {np.mean(params):.1f} ± {np.std(params):.1f}M")
        print(f"     Range: {np.min(params):.1f} - {np.max(params):.1f}M")
        
        # Component analysis
        self._print_component_analysis(results)
    
    def _print_component_analysis(self, results: Dict[str, Any]):
        """Print analysis of component contributions"""
        print(f"\n🔬 Component Contribution Analysis:")
        
        # Baseline performance
        baseline_loss = results.get('baseline', {}).get('val_loss', None)
        if baseline_loss is None:
            print("   ⚠️  Baseline not available for comparison")
            return
        
        print(f"   Baseline (control): {baseline_loss:.4f}")
        
        # Individual component contributions
        individual_components = {
            'rmsnorm': 'DeepSeek RMSNorm',
            'mlp': 'DeepSeek MLP', 
            'moe': 'GLM4 MoE',
            'attention': 'DeepSeek Attention'
        }
        
        print(f"\n   Individual Component Improvements:")
        for comp, desc in individual_components.items():
            if comp in results:
                comp_loss = results[comp]['val_loss']
                improvement = (baseline_loss - comp_loss) / baseline_loss * 100
                print(f"     {desc}: {comp_loss:.4f} ({improvement:+.1f}%)")
        
        # Best combination
        if 'all_components' in results:
            all_loss = results['all_components']['val_loss']
            improvement = (baseline_loss - all_loss) / baseline_loss * 100
            print(f"\n   All Components Combined: {all_loss:.4f} ({improvement:+.1f}%)")
    
    def _create_loss_vs_time_plot(self):
        """Create comprehensive loss vs time visualization"""
        if not self.loss_curves:
            return
        
        plt.figure(figsize=(16, 10))
        
        # Color scheme for different component types
        colors = {
            'baseline': 'black',
            'rmsnorm': 'blue',
            'mlp': 'green', 
            'moe': 'red',
            'attention': 'orange',
            'rmsnorm_mlp': 'purple',
            'rmsnorm_moe': 'brown',
            'mlp_moe': 'pink',
            'attention_rmsnorm': 'gray',
            'attention_mlp': 'olive',
            'attention_moe': 'cyan',
            'all_components': 'magenta'
        }
        
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+']
        
        for i, (name, data) in enumerate(self.loss_curves.items()):
            if 'eval_times' in data and len(data['eval_times']) > 1:
                # Convert timestamps to elapsed time in minutes
                start_time = data['eval_times'][0]
                elapsed_times = [(t - start_time) / 60 for t in data['eval_times']]
                
                color = colors.get(name, 'gray')
                marker = markers[i % len(markers)]
                
                plt.plot(elapsed_times, data['eval_losses'], 
                        color=color, marker=marker, linewidth=2, markersize=6,
                        label=f'{name} (Final: {data["eval_losses"][-1]:.4f})')
        
        plt.xlabel('Time (minutes)', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Experiment 6: Ablation Study - Validation Loss vs Time\n(All Model Configurations)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "exp6_ablation_loss_vs_time_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Comprehensive ablation study plot saved as: {plot_file}")


def main():
    """Main function to run Experiment 6"""
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create configuration for fair comparison (same as exp4/exp5)
    base_config = MoEModelConfig(
        max_steps=1000,  # Full training
        batch_size=16,  # Same batch size as previous experiments
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
    
    print(f"🚀 Experiment 6 Configuration:")
    print(f"   Steps: {base_config.max_steps} (full training)")
    print(f"   Batch Size: {base_config.batch_size}")
    print(f"   Model: {base_config.d_model}d, {base_config.n_layers}L, {base_config.n_heads}H")
    print(f"   MoE: {base_config.num_experts} experts, top-{base_config.expert_top_k}")
    print(f"   Models: {len(ABLATION_MODELS)} ablation configurations")
    print(f"   Expected Training Time: ~2-3 hours total")
    
    # Create trainer
    trainer = Experiment6Trainer(base_config)
    
    # First run 5-step tests to verify all models work
    print(f"\n🧪 Running 5-step tests to verify all models work...")
    test_results = trainer.run_test_experiment()
    
    # Check if all tests passed
    failed_tests = [name for name, result in test_results.items() if "error" in result]
    if failed_tests:
        print(f"\n❌ Some models failed testing: {failed_tests}")
        print("Please fix the issues before running the full experiment.")
        return
    
    print(f"\n✅ All models passed testing! Proceeding with full experiment...")
    
    # Run full experiment
    results = trainer.run_full_experiment()
    
    print(f"\n✅ Experiment 6 completed!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
