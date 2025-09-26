"""
Enhanced Experiment 1: DeepSeek Attention Integration with Comprehensive Metrics

This script trains and compares different attention configurations with:
- Pure baseline (no DeepSeek components)
- DeepSeek variants with LoRA and enhanced features
- 30% increased compute and memory
- Comprehensive metrics: memory, FLOPs, timing, loss tracking
- Loss vs time visualization on same graph
"""

import torch
import time
import json
import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple
from pathlib import Path
from scipy import stats
import torch.profiler
from contextlib import nullcontext

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed
from experiments.exp1_config_import import get_experiment_configs, create_config_from_moe_config
from experiments.exp1_deepseek_import import DeepSeekMoEModel
from models.moe_llm import MoEMinimalLLM


class EnhancedExperiment1Trainer:
    """Enhanced trainer with comprehensive metrics and 30% increased resources"""
    
    def __init__(self, base_config: MoEModelConfig, output_dir: str = "experiments/exp1_enhanced_results"):
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
        
        # Track all experiments for comparison
        self.all_results = {}
        self.all_loss_curves = {}
        self.all_timing_data = {}
    
    def create_model(self, config, variant_name: str):
        """Create model based on configuration"""
        if variant_name == "pure_baseline":
            # Use original MoE model for pure baseline (no DeepSeek)
            return MoEMinimalLLM(self.base_config)
        else:
            # Use DeepSeek MoE model for other variants
            return DeepSeekMoEModel(config, num_experts=8, top_k=2)
    
    def count_parameters(self, model):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def estimate_flops(self, model, batch_size: int, seq_len: int) -> float:
        """Estimate FLOPs for a forward pass"""
        # Rough estimation based on model architecture
        total_params = sum(p.numel() for p in model.parameters())
        
        # Attention FLOPs: O(n^2 * d) per head
        if hasattr(model, 'config'):
            d_model = getattr(model.config, 'hidden_size', getattr(model.config, 'd_model', 768))
            n_heads = getattr(model.config, 'num_attention_heads', getattr(model.config, 'n_heads', 12))
            n_layers = getattr(model.config, 'num_hidden_layers', getattr(model.config, 'n_layers', 10))
        else:
            d_model = 768
            n_heads = 12
            n_layers = 10
        
        # Attention FLOPs per layer
        attn_flops_per_layer = 2 * seq_len * seq_len * d_model + 2 * seq_len * d_model * d_model
        
        # FFN FLOPs per layer (assuming MoE with 2 active experts)
        ffn_flops_per_layer = 2 * seq_len * d_model * (d_model * 4) * 2  # 2 active experts
        
        # Total FLOPs
        total_flops = batch_size * n_layers * (attn_flops_per_layer + ffn_flops_per_layer)
        
        return total_flops
    
    def train_configuration(self, config, variant_name: str) -> Dict[str, Any]:
        """Train a single configuration with comprehensive metrics"""
        print(f"\n{'='*80}")
        print(f"🧪 Training: {variant_name} (Enhanced with 30% more resources)")
        print(f"{'='*80}")
        
        # Set vocab_size for DeepSeek configs, use base_config for pure baseline
        if config is not None:
            config.vocab_size = self.vocab_size
        else:
            # For pure baseline, use base_config directly
            config = self.base_config
        
        # Use the larger batch size from config
        enhanced_batch_size = self.base_config.batch_size
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=enhanced_batch_size, 
            shuffle=True, 
            num_workers=4  # Increased workers
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=enhanced_batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        # Print configuration details
        self._print_config_details(config, variant_name, enhanced_batch_size)
        
        # Count parameters
        model = self.create_model(config, variant_name)
        param_count = self.count_parameters(model)
        print(f"   Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
        
        # Estimate FLOPs
        flops_per_forward = self.estimate_flops(model, enhanced_batch_size, self.base_config.max_seq_len)
        print(f"   Estimated FLOPs per forward: {flops_per_forward/1e9:.2f} GFLOPs")
        
        # Measure memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1e9  # GB
        else:
            initial_memory = psutil.Process().memory_info().rss / 1e9  # GB
        
        # Track training metrics with precise timing
        training_start_time = time.time()
        training_metrics = self._train_with_metrics(model, train_loader, val_loader, variant_name)
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        
        # Measure memory after training
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
            final_memory = torch.cuda.memory_allocated() / 1e9  # GB
            memory_used = peak_memory - initial_memory
        else:
            final_memory = psutil.Process().memory_info().rss / 1e9  # GB
            memory_used = final_memory - initial_memory
            peak_memory = final_memory
        
        # Compile results with precise timing
        results = {
            **training_metrics,
            'initial_memory_gb': initial_memory,
            'peak_memory_gb': peak_memory,
            'final_memory_gb': final_memory,
            'memory_used_gb': memory_used,
            'experiment_name': variant_name,
            'uses_deepseek': variant_name != "pure_baseline",
            'parameter_count': param_count,
            'parameters_millions': param_count / 1e6,
            'flops_per_forward_gflops': flops_per_forward / 1e9,
            'enhanced_batch_size': enhanced_batch_size,
            'total_training_time_seconds': total_training_time,
            'total_training_time_minutes': total_training_time / 60
        }
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _train_with_metrics(self, model, train_loader, val_loader, variant_name: str) -> Dict[str, Any]:
        """Train model with comprehensive metrics tracking"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizers (using existing setup_muon_optimizer)
        from training.trainer import setup_muon_optimizer
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
        
        scaler = torch.amp.GradScaler('cuda') if self.base_config.use_amp else None
        
        # Training loop with metrics
        model.train()
        step = 0
        
        # Track metrics over time
        step_times = []
        step_losses = []
        step_memory = []
        eval_steps = []
        eval_losses = []
        eval_times = []
        
        start_time = time.time()
        
        print(f"🚀 Starting training loop for {self.base_config.max_steps} steps...")
        
        while step < self.base_config.max_steps:
            step_start_time = time.time()
            
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= self.base_config.max_steps:
                    break
                
                x, y = x.to(device), y.to(device)
                
                # Forward pass with timing
                forward_start = time.time()
                if self.base_config.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits, aux_loss = model(x, return_aux_loss=True)
                        ce_loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, self.vocab_size), y.view(-1)
                        )
                        total_loss = ce_loss
                        if aux_loss is not None:
                            total_loss = total_loss + aux_loss
                        loss = total_loss / self.base_config.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, self.vocab_size), y.view(-1)
                    )
                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss
                    loss = total_loss / self.base_config.gradient_accumulation_steps
                    loss.backward()
                
                forward_time = time.time() - forward_start
                
                # Optimizer step
                if (step + 1) % self.base_config.gradient_accumulation_steps == 0:
                    if self.base_config.use_amp:
                        for optimizer in optimizers:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.base_config.grad_clip)
                        
                        for optimizer in optimizers:
                            scaler.step(optimizer)
                            optimizer.zero_grad()
                        for scheduler in schedulers:
                            scheduler.step()
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.base_config.grad_clip)
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        for scheduler in schedulers:
                            scheduler.step()
                
                # Track metrics
                step_time = time.time() - step_start_time
                step_times.append(step_time)
                step_losses.append(ce_loss.item())
                
                if torch.cuda.is_available():
                    step_memory.append(torch.cuda.memory_allocated() / 1e9)
                
                # Progress logging every 50 steps for longer training
                if step % 50 == 0:
                    elapsed_time = time.time() - start_time
                    steps_per_sec = step / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (self.base_config.max_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                    eta_minutes = eta_seconds / 60
                    print(f"Step {step}/{self.base_config.max_steps}: Loss={ce_loss.item():.4f}, "
                          f"Time={step_time*1000:.1f}ms, Memory={torch.cuda.memory_allocated()/1e9:.2f}GB, "
                          f"ETA={eta_minutes:.1f}min")
                
                # Evaluation
                if step % self.base_config.eval_every == 0 and step > 0:
                    print(f"\n🔍 Evaluating model at step {step}...")
                    eval_start_time = time.time()
                    eval_metrics = self._evaluate_model(model, val_loader)
                    eval_time = time.time() - eval_start_time
                    
                    eval_steps.append(step)
                    eval_losses.append(eval_metrics['val_loss'])
                    eval_times.append(time.time())
                    
                    print(f"✅ Evaluation complete in {eval_time:.2f}s:")
                    print(f"   Val Loss: {eval_metrics['val_loss']:.4f}")
                    print(f"   Val Acc: {eval_metrics['val_accuracy']:.4f}")
                    print(f"   Val PPL: {eval_metrics['val_perplexity']:.2f}")
                    print(f"   Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    print()
                
                step += 1
                step_start_time = time.time()
        
        total_time = time.time() - start_time
        
        # Final evaluation
        print(f"\n🔍 Final evaluation at step {self.base_config.max_steps}...")
        final_eval = self._evaluate_model(model, val_loader)
        eval_steps.append(self.base_config.max_steps)
        eval_losses.append(final_eval['val_loss'])
        eval_times.append(time.time())
        print(f"✅ Final evaluation complete:")
        print(f"   Val Loss: {final_eval['val_loss']:.4f}")
        print(f"   Val Acc: {final_eval['val_accuracy']:.4f}")
        print(f"   Val PPL: {final_eval['val_perplexity']:.2f}")
        print()
        
        # Store loss curve data
        self.all_loss_curves[variant_name] = {
            'eval_steps': eval_steps,
            'eval_losses': eval_losses,
            'eval_times': eval_times,
            'step_times': step_times,
            'step_losses': step_losses,
            'step_memory': step_memory
        }
        
        return {
            **final_eval,
            'training_time_minutes': total_time / 60,
            'avg_step_time_ms': np.mean(step_times) * 1000,
            'avg_forward_time_ms': np.mean([t for t in step_times]) * 1000,
            'peak_step_memory_gb': np.max(step_memory) if step_memory else 0,
            'avg_step_memory_gb': np.mean(step_memory) if step_memory else 0
        }
    
    def _evaluate_model(self, model, val_loader):
        """Evaluate model on validation set (limited to first 10 batches for speed)"""
        model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        # Limit evaluation to first 10 batches for speed
        max_eval_batches = 10
        batch_count = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                if batch_count >= max_eval_batches:
                    break
                    
                x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                
                if hasattr(model, 'forward') and 'return_aux_loss' in model.forward.__code__.co_varnames:
                    logits, aux_loss = model(x, return_aux_loss=True)
                else:
                    logits = model(x)
                
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
    
    def _print_config_details(self, config, variant_name: str, batch_size: int):
        """Print detailed configuration information"""
        print(f"\n📋 Configuration Details:")
        if hasattr(config, 'hidden_size'):
            print(f"   Architecture: {config.hidden_size}d, {config.num_hidden_layers}L, {config.num_attention_heads}H, {config.intermediate_size}ff")
        else:
            print(f"   Architecture: {self.base_config.d_model}d, {self.base_config.n_layers}L, {self.base_config.n_heads}H, {self.base_config.d_ff}ff")
        print(f"   MoE: 8 experts, top-2 routing")
        print(f"   Training: {self.base_config.max_steps} steps, batch size {batch_size}")
        
        # DeepSeek specific details
        if variant_name != "pure_baseline":
            print(f"   Using DeepSeek Components: ✅")
            if hasattr(config, 'q_lora_rank') and config.q_lora_rank is not None:
                print(f"   Q LoRA rank: {config.q_lora_rank}")
            if hasattr(config, 'kv_lora_rank') and config.kv_lora_rank is not None:
                print(f"   KV LoRA rank: {config.kv_lora_rank}")
            if hasattr(config, '_attn_implementation') and config._attn_implementation == "flash_attention_2":
                print(f"   Flash Attention: Enabled")
            if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                print(f"   RoPE Scaling: {config.rope_scaling}")
            if hasattr(config, 'attention_bias') and config.attention_bias:
                print(f"   Attention Bias: Enabled")
        else:
            print(f"   Using DeepSeek Components: ❌ (pure baseline)")
    
    def _print_results(self, results: Dict[str, Any]):
        """Print detailed results"""
        print(f"\n🎯 Results for {results['experiment_name']}:")
        print(f"⏱️ Training time: {results['total_training_time_minutes']:.2f} minutes ({results['total_training_time_seconds']:.1f} seconds)")
        print(f"💾 Memory usage:")
        print(f"   Initial: {results['initial_memory_gb']:.2f} GB")
        print(f"   Peak: {results['peak_memory_gb']:.2f} GB")
        print(f"   Final: {results['final_memory_gb']:.2f} GB")
        print(f"   Used: {results['memory_used_gb']:.2f} GB")
        print(f"   Avg Step: {results['avg_step_memory_gb']:.2f} GB")
        print(f"🏆 Final Results:")
        print(f"   Validation Loss: {results['val_loss']:.4f}")
        print(f"   Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"   Validation Perplexity: {results['val_perplexity']:.2f}")
        print(f"   Parameters: {results['parameter_count']:,} ({results['parameters_millions']:.2f}M)")
        print(f"   FLOPs per forward: {results['flops_per_forward_gflops']:.2f} GFLOPs")
        print(f"   Avg step time: {results['avg_step_time_ms']:.1f} ms")
    
    def run_experiment(self, configs_to_run: List[str] = None) -> Dict[str, Any]:
        """Run the full experiment with multiple configurations"""
        if configs_to_run is None:
            configs_to_run = ["pure_baseline", "lora", "enhanced"]
        
        # Get experiment configurations
        experiment_configs = get_experiment_configs()
        
        # Add pure baseline config
        experiment_configs["pure_baseline"] = None  # Will use base_config directly
        
        # Filter configurations to run
        configs_to_run = {name: experiment_configs[name] for name in configs_to_run 
                         if name in experiment_configs}
        
        print(f"\n🚀 Starting Enhanced Experiment 1: DeepSeek Attention Integration")
        print(f"📋 Running {len(configs_to_run)} configurations: {list(configs_to_run.keys())}")
        print(f"💪 10x longer training with precise timing measurements")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Run each configuration
        results = {}
        for name, config in configs_to_run.items():
            try:
                # Clear GPU memory before each experiment
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                results[name] = self.train_configuration(config, name)
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                results[name] = {"error": str(e)}
                
                # Clear memory after error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Store results
        self.all_results = results
        
        # Save results
        self._save_results(results)
        
        # Print comparison
        self._print_comparison(results)
        
        # Create loss vs time visualization
        self._create_loss_vs_time_plot()
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results to file"""
        results_file = self.output_dir / "enhanced_experiment1_results.json"
        
        # Convert any non-serializable objects to strings
        serializable_results = {}
        for name, result in results.items():
            serializable_results[name] = {}
            for key, value in result.items():
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_results[name][key] = value
                except (TypeError, ValueError):
                    serializable_results[name][key] = str(value)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def _print_comparison(self, results: Dict[str, Any]):
        """Print comprehensive comparison of all configurations"""
        print(f"\n{'='*100}")
        print(f"📊 ENHANCED EXPERIMENT 1 COMPARISON: DeepSeek Attention Integration (10x Longer Training)")
        print(f"{'='*100}")
        
        # Create comparison table
        print(f"{'Config':<15} {'Val Loss':<10} {'Val Acc':<10} {'Val Perp':<10} {'Time (min)':<12} {'Peak Mem (GB)':<15} {'Params (M)':<12} {'FLOPs (G)':<12} {'DeepSeek':<10}")
        print(f"{'-'*130}")
        
        for name, result in results.items():
            if "error" not in result:
                val_loss = result.get('val_loss', 0)
                val_acc = result.get('val_accuracy', 0)
                val_perp = result.get('val_perplexity', 0)
                time_min = result.get('total_training_time_minutes', 0)  # Use precise timing
                peak_mem = result.get('peak_memory_gb', 0)
                params_m = result.get('parameters_millions', 0)
                flops_g = result.get('flops_per_forward_gflops', 0)
                uses_deepseek = "✅" if result.get('uses_deepseek', False) else "❌"
                
                print(f"{name:<15} {val_loss:<10.4f} {val_acc:<10.4f} {val_perp:<10.2f} {time_min:<12.2f} {peak_mem:<15.2f} {params_m:<12.2f} {flops_g:<12.2f} {uses_deepseek:<10}")
            else:
                print(f"{name:<15} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<10}")
        
        print(f"{'-'*130}")
        
        # Statistical analysis
        self._print_statistical_analysis(results)
        
        # Find best configuration
        valid_results = {name: result for name, result in results.items() 
                        if "error" not in result}
        
        if valid_results:
            best_loss = min(valid_results.items(), key=lambda x: x[1].get('val_loss', float('inf')))
            best_acc = max(valid_results.items(), key=lambda x: x[1].get('val_accuracy', 0))
            best_perp = min(valid_results.items(), key=lambda x: x[1].get('val_perplexity', float('inf')))
            fastest = min(valid_results.items(), key=lambda x: x[1].get('total_training_time_minutes', float('inf')))
            most_efficient = min(valid_results.items(), key=lambda x: x[1].get('peak_memory_gb', float('inf')))
            
            print(f"\n🏆 Best Results:")
            print(f"   Lowest Loss: {best_loss[0]} ({best_loss[1]['val_loss']:.4f})")
            print(f"   Highest Accuracy: {best_acc[0]} ({best_acc[1]['val_accuracy']:.4f})")
            print(f"   Lowest Perplexity: {best_perp[0]} ({best_perp[1]['val_perplexity']:.2f})")
            print(f"   Fastest Training: {fastest[0]} ({fastest[1]['total_training_time_minutes']:.2f} min)")
            print(f"   Most Memory Efficient: {most_efficient[0]} ({most_efficient[1]['peak_memory_gb']:.2f} GB)")
        
        print(f"{'='*130}")
    
    def _print_statistical_analysis(self, results: Dict[str, Any]):
        """Print statistical analysis of results"""
        valid_results = {name: result for name, result in results.items() 
                        if "error" not in result}
        
        if len(valid_results) < 2:
            return
        
        print(f"\n📈 Statistical Analysis:")
        
        # Extract metrics
        losses = [result['val_loss'] for result in valid_results.values()]
        times = [result['total_training_time_minutes'] for result in valid_results.values()]  # Use precise timing
        params = [result['parameters_millions'] for result in valid_results.values()]
        flops = [result['flops_per_forward_gflops'] for result in valid_results.values()]
        
        print(f"   Loss Statistics:")
        print(f"     Mean: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"     Range: {np.min(losses):.4f} - {np.max(losses):.4f}")
        
        # Effect size for best vs baseline
        if 'pure_baseline' in valid_results and len(valid_results) > 1:
            baseline_loss = valid_results['pure_baseline']['val_loss']
            best_loss = min(losses)
            effect_size = abs(best_loss - baseline_loss) / np.std(losses)
            print(f"     Effect Size (best vs baseline): {effect_size:.3f}")
            if effect_size < 0.2:
                print(f"     Interpretation: Negligible effect")
            elif effect_size < 0.5:
                print(f"     Interpretation: Small effect")
            elif effect_size < 0.8:
                print(f"     Interpretation: Medium effect")
            else:
                print(f"     Interpretation: Large effect")
        
        print(f"   Efficiency Analysis:")
        print(f"     Parameter Range: {np.min(params):.2f}M - {np.max(params):.2f}M")
        print(f"     Time Range: {np.min(times):.1f} - {np.max(times):.1f} min")
        print(f"     FLOPs Range: {np.min(flops):.2f} - {np.max(flops):.2f} GFLOPs")
        
        # Parameter efficiency (loss per million parameters)
        param_efficiency = [(result['val_loss'], result['parameters_millions'], name) for name, result in valid_results.items()]
        param_efficiency.sort(key=lambda x: x[0] / x[1])  # Sort by loss/param ratio
        
        print(f"   Parameter Efficiency (loss per M params):")
        for loss, param, name in param_efficiency:
            efficiency = loss / param
            print(f"     {name}: {efficiency:.4f}")
    
    def _create_loss_vs_time_plot(self):
        """Create loss vs time visualization on same graph"""
        if not self.all_loss_curves:
            return
        
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, (name, data) in enumerate(self.all_loss_curves.items()):
            if 'eval_times' in data and len(data['eval_times']) > 1:
                # Convert timestamps to elapsed time in minutes
                start_time = data['eval_times'][0]
                elapsed_times = [(t - start_time) / 60 for t in data['eval_times']]
                
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                plt.plot(elapsed_times, data['eval_losses'], 
                        color=color, marker=marker, linewidth=2, markersize=6,
                        label=f'{name} (Final: {data["eval_losses"][-1]:.4f})')
        
        plt.xlabel('Time (minutes)', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Validation Loss vs Time - Enhanced Experiment 1\n(10x Longer Training)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "loss_vs_time_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Loss vs time comparison plot saved as: {plot_file}")


def main():
    """Main function to run Enhanced Experiment 1"""
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create enhanced base configuration with 10x longer training
    base_config = MoEModelConfig(
        max_steps=1000,  # 10x longer training
        batch_size=32,  # Larger batch size for faster training
        max_tokens=2000000,  # Keep reasonable data size
        eval_every=100,  # Evaluate every 100 steps for longer training
        num_documents=8000,  # Keep reasonable data size
        max_seq_len=512,
        d_model=512,  # Reduced from 768
        n_heads=8,    # Reduced from 12
        n_layers=6,   # Reduced from 10
        d_ff=2048,    # Reduced from 3072
    )
    
    print(f"🚀 ENHANCED Experiment Configuration (10x Longer Training):")
    print(f"   Steps: {base_config.max_steps} (10x longer training)")
    print(f"   Batch Size: {base_config.batch_size} (larger for faster training)")
    print(f"   Tokens: {base_config.max_tokens:,}")
    print(f"   Model: {base_config.d_model}d, {base_config.n_layers}L, {base_config.n_heads}H, {base_config.d_ff}ff")
    print(f"   Sequence Length: {base_config.max_seq_len}")
    print(f"   Evaluation: Every {base_config.eval_every} steps")
    print(f"   Expected Memory Usage: ~8-12 GB (reduced model size)")
    print(f"   Expected Training Time: ~4-5 minutes per configuration")
    
    # Create trainer
    trainer = EnhancedExperiment1Trainer(base_config)
    
    # Run experiment with all configurations
    results = trainer.run_experiment(['pure_baseline', 'lora', 'enhanced'])
    
    print(f"\n✅ Enhanced Experiment 1 completed!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
