"""
Experiment 2: Fair Architecture Search (Fixed Model Size)

This script performs a fair architecture search by keeping model size constant
and testing different attention mechanisms to isolate architectural differences.

Features:
- Fixed model size: 512d, 8L, 8H, 2048ff (fair comparison)
- 13 attention mechanisms: baseline, lora variants, enhanced variants, rope variants, bias_only
- Fast training with 50 steps for rapid comparison
- Comprehensive metrics tracking
- Single comprehensive visualization
- Fair comparison with memory/time tracking
"""

import torch
import time
import json
import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional dependency
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from scipy import stats
import torch.profiler
from contextlib import nullcontext
# import pandas as pd  # Optional dependency
from itertools import product

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed
from experiments.exp2_config_import import (
    get_architecture_search_configs, 
    get_training_configs,
    create_moe_config_from_architecture,
    create_config_from_moe_config
)
from experiments.exp1_deepseek_import import DeepSeekMoEModel
from models.moe_llm import MoEMinimalLLM


class ArchitectureSearchTrainer:
    """Fair architecture search trainer with fixed model size"""
    
    def __init__(self, output_dir: str = "experiments/exp2_architecture_search_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track all experiments
        self.all_results = {}
        self.all_loss_curves = {}
        self.all_timing_data = {}
        self.experiment_metadata = {}
        
        # Load data once for all experiments
        self._load_data()
        
        # Set vocab size from tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        
        print(f"🔍 Architecture Search Trainer initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Dataset: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")
    
    def _load_data(self):
        """Load data once for all experiments"""
        # Use medium config for data loading (will be overridden per experiment)
        base_config = create_moe_config_from_architecture("medium", "medium")
        
        self.texts, self.tokenizer, self.tokens = load_and_cache_data(base_config)
        # Vocab size will be set from tokenizer after loading
        
        # Create dataset
        self.dataset = TextTokenDataset(self.tokens, base_config.max_seq_len)
        
        # Train/val split
        val_size = len(self.dataset) // 10
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    
    def create_model(self, config, variant_name: str, moe_config: MoEModelConfig):
        """Create model based on configuration"""
        if "baseline" in variant_name:
            # Use original MoE model for baseline configurations
            return MoEMinimalLLM(moe_config)
        else:
            # Use DeepSeek MoE model for other variants
            return DeepSeekMoEModel(config, num_experts=8, top_k=2)
    
    def count_parameters(self, model):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def estimate_flops(self, model, batch_size: int, seq_len: int) -> float:
        """Estimate FLOPs for a forward pass"""
        try:
            # Create dummy input
            dummy_input = torch.randint(0, self.vocab_size, (batch_size, seq_len))
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Rough FLOP estimation: 2 * parameters * sequence_length
            # This is a simplified estimate
            flops = 2 * total_params * seq_len
            
            return flops / 1e9  # Convert to GFLOPs
        except Exception as e:
            print(f"Warning: Could not estimate FLOPs: {e}")
            return 0.0
    
    def train_configuration(self, config, variant_name: str, moe_config: MoEModelConfig, 
                          training_mode: str = "medium") -> Dict[str, Any]:
        """Train a single configuration with comprehensive metrics"""
        
        print(f"\n🚀 Training {variant_name} ({training_mode} mode)")
        print(f"   Model: {moe_config.d_model}d, {moe_config.n_layers}L, {moe_config.n_heads}H, {moe_config.d_ff}ff")
        print(f"   Training: {moe_config.max_steps} steps, batch size {moe_config.batch_size}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1e9
        
        # Track timing
        start_time = time.time()
        step_times = []
        forward_times = []
        memory_usage = []
        
        try:
            # Create model
            model = self.create_model(config, variant_name, moe_config)
            
            if torch.cuda.is_available():
                model = model.cuda()
                model_memory = torch.cuda.memory_allocated() / 1e9
                print(f"   Model memory: {model_memory:.2f} GB")
            
            # Count parameters
            param_count = self.count_parameters(model)
            param_millions = param_count / 1e6
            
            # Estimate FLOPs
            flops = self.estimate_flops(model, moe_config.batch_size, moe_config.max_seq_len)
            
            # Update config with correct vocab size
            config.vocab_size = self.vocab_size
            
            # Create data loaders
            train_loader = DataLoader(
                self.train_dataset, 
                batch_size=moe_config.batch_size, 
                shuffle=True,
                num_workers=0
            )
            val_loader = DataLoader(
                self.val_dataset, 
                batch_size=moe_config.batch_size, 
                shuffle=False,
                num_workers=0
            )
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Training loop with metrics
            model.train()
            loss_curve = []
            val_losses = []
            val_accuracies = []
            val_perplexities = []
            
            step_count = 0
            for epoch in range(moe_config.max_steps):
                for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
                    if step_count >= moe_config.max_steps:
                        break
                    
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        target_ids = target_ids.cuda()
                    
                    # Forward pass timing
                    forward_start = time.time()
                    optimizer.zero_grad()
                    
                    outputs = model(input_ids)
                    
                    # Handle tuple output (logits, aux_loss)
                    if isinstance(outputs, tuple):
                        logits, aux_loss = outputs
                        if aux_loss is not None:
                            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1)) + aux_loss
                        else:
                            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    else:
                        loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                    
                    forward_time = (time.time() - forward_start) * 1000
                    forward_times.append(forward_time)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Step timing
                    step_time = (time.time() - forward_start) * 1000
                    step_times.append(step_time)
                    
                    # Memory tracking
                    if torch.cuda.is_available():
                        memory_usage.append(torch.cuda.memory_allocated() / 1e9)
                    
                    loss_curve.append(loss.item())
                    step_count += 1
                    
                    # Evaluation
                    if step_count % moe_config.eval_every == 0:
                        val_loss, val_acc, val_perp = self.evaluate_model(model, val_loader, criterion)
                        val_losses.append(val_loss)
                        val_accuracies.append(val_acc)
                        val_perplexities.append(val_perp)
                        
                        print(f"   Step {step_count}: Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
                    
                    if step_count >= moe_config.max_steps:
                        break
            
            # Final evaluation
            final_val_loss, final_val_acc, final_val_perp = self.evaluate_model(model, val_loader, criterion)
            
            # Calculate metrics
            total_time = time.time() - start_time
            avg_step_time = np.mean(step_times) if step_times else 0
            avg_forward_time = np.mean(forward_times) if forward_times else 0
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                final_memory = torch.cuda.memory_allocated() / 1e9
                memory_used = peak_memory - initial_memory
            else:
                peak_memory = 0
                final_memory = 0
                memory_used = 0
            
            # Determine if uses DeepSeek
            uses_deepseek = "baseline" not in variant_name
            
            results = {
                "val_loss": final_val_loss,
                "val_accuracy": final_val_acc,
                "val_perplexity": final_val_perp,
                "training_time_minutes": total_time / 60,
                "avg_step_time_ms": avg_step_time,
                "avg_forward_time_ms": avg_forward_time,
                "peak_step_memory_gb": max(memory_usage) if memory_usage else 0,
                "avg_step_memory_gb": np.mean(memory_usage) if memory_usage else 0,
                "initial_memory_gb": initial_memory if torch.cuda.is_available() else 0,
                "peak_memory_gb": peak_memory,
                "final_memory_gb": final_memory,
                "memory_used_gb": memory_used,
                "experiment_name": variant_name,
                "uses_deepseek": uses_deepseek,
                "parameter_count": param_count,
                "parameters_millions": param_millions,
                "flops_per_forward_gflops": flops,
                "training_mode": training_mode,
                "model_size": moe_config.d_model,
                "num_layers": moe_config.n_layers,
                "num_heads": moe_config.n_heads,
                "d_ff": moe_config.d_ff,
                "total_training_time_seconds": total_time,
                "total_training_time_minutes": total_time / 60,
                "loss_curve": loss_curve,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
                "val_perplexities": val_perplexities,
            }
            
            print(f"✅ {variant_name} completed:")
            print(f"   Val Loss: {final_val_loss:.4f}")
            print(f"   Val Acc: {final_val_acc:.4f}")
            print(f"   Val Perp: {final_val_perp:.2f}")
            print(f"   Time: {total_time/60:.2f} min")
            print(f"   Memory: {memory_used:.2f} GB")
            print(f"   Params: {param_millions:.1f}M")
            
            return results
            
        except Exception as e:
            print(f"❌ Error training {variant_name}: {e}")
            return {
                "error": str(e),
                "experiment_name": variant_name,
                "training_mode": training_mode,
                "uses_deepseek": "baseline" not in variant_name,
            }
        
        finally:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def evaluate_model(self, model, val_loader, criterion):
        """Evaluate model on validation set (limited to first 10 batches for speed)"""
        model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        # Limit evaluation to first 10 batches for speed
        max_eval_batches = 10
        batch_count = 0
        
        print(f"   🔍 Running validation (limited to {max_eval_batches} batches)...")
        
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                if batch_count >= max_eval_batches:
                    break
                    
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    target_ids = target_ids.cuda()
                
                outputs = model(input_ids)
                
                # Handle tuple output (logits, aux_loss)
                if isinstance(outputs, tuple):
                    logits, aux_loss = outputs
                    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    predictions = torch.argmax(logits, dim=-1)
                else:
                    loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                    predictions = torch.argmax(outputs, dim=-1)
                
                total_loss += loss.item() * input_ids.size(0) * input_ids.size(1)
                
                # Calculate accuracy
                correct = (predictions == target_ids).sum().item()
                total_correct += correct
                total_tokens += target_ids.numel()
                
                batch_count += 1
        
        model.train()
        
        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        perplexity = np.exp(min(avg_loss, 20))  # Cap to avoid overflow
        
        print(f"   ✅ Validation complete: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, Perp={perplexity:.2f}")
        
        return avg_loss, accuracy, perplexity
    
    def run_architecture_search(self, training_mode: str = "fast", 
                              selected_configs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run fair architecture search with fixed model size"""
        
        print(f"\n🚀 Starting Architecture Search (Mode: {training_mode})")
        
        # Get all configurations
        all_configs = get_architecture_search_configs()
        
        # Filter configurations if specified
        if selected_configs:
            configs_to_run = {name: config for name, config in all_configs.items() 
                            if name in selected_configs}
        else:
            configs_to_run = all_configs
        
        print(f"📋 Running {len(configs_to_run)} configurations")
        print(f"🔧 Training mode: {training_mode}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Run each configuration
        results = {}
        for i, (name, config) in enumerate(configs_to_run.items()):
            try:
                print(f"\n📊 Progress: {i+1}/{len(configs_to_run)} - {name}")
                
                # All configurations use the same model size for fair comparison
                size_name = "medium"  # Fixed size for all configs
                
                # Create MoE config
                moe_config = create_moe_config_from_architecture(size_name, training_mode)
                
                # Set correct vocab size from tokenizer
                moe_config.vocab_size = self.vocab_size
                
                # Train configuration
                result = self.train_configuration(config, name, moe_config, training_mode)
                results[name] = result
                
                # Store metadata
                self.experiment_metadata[name] = {
                    "model_size": "medium",  # Fixed size for all
                    "attention_type": '_'.join(name.split('_')[1:]),
                    "training_mode": training_mode,
                    "config_index": i
                }
                
            except Exception as e:
                print(f"❌ Error with {name}: {e}")
                results[name] = {"error": str(e), "experiment_name": name}
        
        # Store results
        self.all_results = results
        
        # Save results
        self._save_results(results, training_mode)
        
        # Print summary
        self._print_summary(results)
        
        # Create comprehensive visualization
        self._create_comprehensive_plot(results, training_mode)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], training_mode: str):
        """Save results to JSON file"""
        results_file = self.output_dir / f"architecture_search_results_{training_mode}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print summary of results"""
        print(f"\n📊 Architecture Search Summary")
        print(f"=" * 80)
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not successful_results:
            print("❌ No successful experiments")
            return
        
        # Sort by validation loss
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]["val_loss"])
        
        print(f"🏆 Top 5 Configurations by Validation Loss:")
        for i, (name, result) in enumerate(sorted_results[:5]):
            print(f"   {i+1}. {name}: Loss={result['val_loss']:.4f}, "
                  f"Acc={result['val_accuracy']:.4f}, "
                  f"Time={result['training_time_minutes']:.1f}min, "
                  f"Memory={result['memory_used_gb']:.1f}GB")
        
        # Statistics
        losses = [r["val_loss"] for r in successful_results.values()]
        accuracies = [r["val_accuracy"] for r in successful_results.values()]
        times = [r["training_time_minutes"] for r in successful_results.values()]
        memories = [r["memory_used_gb"] for r in successful_results.values()]
        
        print(f"\n📈 Statistics:")
        print(f"   Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f} (range: {np.min(losses):.4f}-{np.max(losses):.4f})")
        print(f"   Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"   Time: {np.mean(times):.1f} ± {np.std(times):.1f} min")
        print(f"   Memory: {np.mean(memories):.1f} ± {np.std(memories):.1f} GB")
    
    def _create_comprehensive_plot(self, results: Dict[str, Any], training_mode: str):
        """Create comprehensive visualization of all results"""
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not successful_results:
            print("❌ No successful results to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Architecture Search Results ({training_mode} mode)', fontsize=16, fontweight='bold')
        
        # Prepare data
        names = list(successful_results.keys())
        losses = [successful_results[name]["val_loss"] for name in names]
        accuracies = [successful_results[name]["val_accuracy"] for name in names]
        perplexities = [successful_results[name]["val_perplexity"] for name in names]
        times = [successful_results[name]["training_time_minutes"] for name in names]
        memories = [successful_results[name]["memory_used_gb"] for name in names]
        params = [successful_results[name]["parameters_millions"] for name in names]
        
        # Color by attention type (since model size is fixed)
        colors = []
        for name in names:
            if "baseline" in name:
                colors.append("red")
            elif "lora" in name:
                colors.append("blue")
            elif "enhanced" in name:
                colors.append("green")
            elif "rope" in name:
                colors.append("orange")
            elif "bias" in name:
                colors.append("purple")
            else:
                colors.append("gray")
        
        # 1. Loss vs Parameters
        axes[0, 0].scatter(params, losses, c=colors, alpha=0.7, s=60)
        axes[0, 0].set_xlabel('Parameters (Millions)')
        axes[0, 0].set_ylabel('Validation Loss')
        axes[0, 0].set_title('Loss vs Parameters (Fixed Model Size)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Accuracy vs Parameters
        axes[0, 1].scatter(params, accuracies, c=colors, alpha=0.7, s=60)
        axes[0, 1].set_xlabel('Parameters (Millions)')
        axes[0, 1].set_ylabel('Validation Accuracy')
        axes[0, 1].set_title('Accuracy vs Parameters (Fixed Model Size)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Time vs Memory
        axes[0, 2].scatter(memories, times, c=colors, alpha=0.7, s=60)
        axes[0, 2].set_xlabel('Memory Usage (GB)')
        axes[0, 2].set_ylabel('Training Time (min)')
        axes[0, 2].set_title('Time vs Memory Usage')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Loss vs Time
        axes[1, 0].scatter(times, losses, c=colors, alpha=0.7, s=60)
        axes[1, 0].set_xlabel('Training Time (min)')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Loss vs Training Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Perplexity vs Parameters
        axes[1, 1].scatter(params, perplexities, c=colors, alpha=0.7, s=60)
        axes[1, 1].set_xlabel('Parameters (Millions)')
        axes[1, 1].set_ylabel('Validation Perplexity')
        axes[1, 1].set_title('Perplexity vs Parameters (Fixed Model Size)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Attention mechanism comparison (box plot)
        attention_types = ["baseline", "lora", "enhanced", "rope", "bias"]
        attn_losses = {attn: [] for attn in attention_types}
        
        for name, result in successful_results.items():
            attn_type = name.split('_')[1] if '_' in name else "baseline"
            # Group similar types
            if "lora" in attn_type:
                attn_losses["lora"].append(result["val_loss"])
            elif "enhanced" in attn_type:
                attn_losses["enhanced"].append(result["val_loss"])
            elif "rope" in attn_type:
                attn_losses["rope"].append(result["val_loss"])
            elif "bias" in attn_type:
                attn_losses["bias"].append(result["val_loss"])
            else:
                attn_losses["baseline"].append(result["val_loss"])
        
        box_data = [attn_losses[attn] for attn in attention_types if attn_losses[attn]]
        box_labels = [attn for attn in attention_types if attn_losses[attn]]
        
        if box_data:
            axes[1, 2].boxplot(box_data, labels=box_labels)
            axes[1, 2].set_xlabel('Attention Mechanism')
            axes[1, 2].set_ylabel('Validation Loss')
            axes[1, 2].set_title('Loss Distribution by Attention Type')
            axes[1, 2].grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Baseline'),
            Patch(facecolor='blue', label='LoRA'),
            Patch(facecolor='green', label='Enhanced'),
            Patch(facecolor='orange', label='RoPE'),
            Patch(facecolor='purple', label='Bias')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"architecture_search_comprehensive_{training_mode}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Comprehensive plot saved as: {plot_file}")


def main():
    """Main function to run Fair Architecture Search Experiment 2"""
    
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create trainer
    trainer = ArchitectureSearchTrainer()
    
    # Get all configurations
    all_configs = get_architecture_search_configs()
    print(f"📋 Total configurations available: {len(all_configs)}")
    
    # Show configuration breakdown
    print(f"🏗️  Model size: Fixed at 512d, 8L, 8H, 2048ff (fair comparison)")
    attention_types = ["baseline", "lora_small", "lora_medium", "lora_large", "lora_xl",
                      "enhanced_small", "enhanced_medium", "enhanced_large", "enhanced_xl",
                      "rope_only", "rope_small", "rope_large", "bias_only"]
    
    print(f"🧠 Attention types: {attention_types}")
    print(f"🔢 Total configurations: {len(all_configs)} (all same model size)")
    
    # Training modes
    training_configs = get_training_configs()
    print(f"\n⚙️  Training modes:")
    for mode, config in training_configs.items():
        print(f"   {mode}: {config['max_steps']} steps, batch size {config['batch_size']} - {config['description']}")
    
    # Run experiment
    print(f"\n🚀 Starting Fair Architecture Search Experiment 2")
    
    # Run fast training for all configurations
    results_fast = trainer.run_architecture_search("fast")
    
    print(f"\n✅ Fair Architecture Search Experiment 2 completed!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
