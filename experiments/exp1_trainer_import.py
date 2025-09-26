"""
Training script for Experiment 1: DeepSeek Attention Integration (Using Original Implementation)

This script trains and compares different attention configurations using the original
DeepSeek attention components from deepseek_modeling.py.
"""

import torch
import time
import json
import os
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from pathlib import Path

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from data.dataset import TextTokenDataset
from training.trainer import train_moe_model
from utils.helpers import set_seed
from .exp1_config_import import get_experiment_configs, create_config_from_moe_config
from .exp1_deepseek_import import DeepSeekMoEModel
from models.moe_llm import MoEMinimalLLM


class Experiment1ImportTrainer:
    """Trainer for Experiment 1 using original DeepSeek components"""
    
    def __init__(self, base_config: MoEModelConfig, output_dir: str = "experiments/exp1_import_results"):
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
    
    def create_model(self, config, variant_name: str):
        """Create model based on configuration"""
        if variant_name == "baseline":
            # Use original MoE model for baseline
            return MoEMinimalLLM(self.base_config)
        else:
            # Use DeepSeek MoE model for other variants
            return DeepSeekMoEModel(config, num_experts=8, top_k=2)
    
    def train_configuration(self, config, variant_name: str) -> Dict[str, Any]:
        """Train a single configuration and return results"""
        print(f"\n{'='*60}")
        print(f"🧪 Training: {variant_name} (using original DeepSeek components)")
        print(f"{'='*60}")
        
        # Set vocab_size
        config.vocab_size = self.vocab_size
        
        # Create data loaders
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
        
        # Print configuration details
        self._print_config_details(config, variant_name)
        
        # Train model
        start_time = time.time()
        model, final_metrics = train_moe_model(self.base_config, train_loader, val_loader)
        total_time = time.time() - start_time
        
        # Add timing information
        final_metrics['training_time_minutes'] = total_time / 60
        final_metrics['experiment_name'] = variant_name
        final_metrics['uses_deepseek'] = variant_name != "baseline"
        
        # Print results
        print(f"\n🎯 Results for {variant_name}:")
        print(f"⏱️ Training time: {total_time/60:.1f} minutes")
        print(f"🏆 Final Results:")
        print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
        print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
        
        return final_metrics
    
    def _print_config_details(self, config, variant_name: str):
        """Print detailed configuration information"""
        print(f"\n📋 Configuration Details:")
        print(f"   Architecture: {config.hidden_size}d, {config.num_hidden_layers}L, {config.num_attention_heads}H, {config.intermediate_size}ff")
        print(f"   MoE: 8 experts, top-2 routing")
        print(f"   Training: {self.base_config.max_steps} steps, batch size {self.base_config.batch_size}")
        
        # DeepSeek specific details
        if variant_name != "baseline":
            print(f"   Using Original DeepSeek Components: ✅")
            if config.q_lora_rank is not None:
                print(f"   Q LoRA rank: {config.q_lora_rank}")
            if config.kv_lora_rank is not None:
                print(f"   KV LoRA rank: {config.kv_lora_rank}")
            if config._attn_implementation == "flash_attention_2":
                print(f"   Flash Attention: Enabled")
            if config.rope_scaling is not None:
                print(f"   RoPE Scaling: {config.rope_scaling}")
            if config.attention_bias:
                print(f"   Attention Bias: Enabled")
        else:
            print(f"   Using Original DeepSeek Components: ❌ (baseline)")
    
    def run_experiment(self, configs_to_run: List[str] = None) -> Dict[str, Any]:
        """Run the full experiment with multiple configurations"""
        if configs_to_run is None:
            configs_to_run = ["baseline", "lora", "flash", "rope", "full"]
        
        # Get experiment configurations
        experiment_configs = get_experiment_configs()
        
        # Filter configurations to run
        configs_to_run = {name: experiment_configs[name] for name in configs_to_run 
                         if name in experiment_configs}
        
        print(f"\n🚀 Starting Experiment 1: DeepSeek Attention Integration (Original Implementation)")
        print(f"📋 Running {len(configs_to_run)} configurations: {list(configs_to_run.keys())}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Run each configuration
        results = {}
        for name, config in configs_to_run.items():
            try:
                results[name] = self.train_configuration(config, name)
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                results[name] = {"error": str(e)}
        
        # Save results
        self._save_results(results)
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results to file"""
        results_file = self.output_dir / "experiment1_import_results.json"
        
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
        """Print comparison of all configurations"""
        print(f"\n{'='*80}")
        print(f"📊 EXPERIMENT 1 COMPARISON: DeepSeek Attention Integration (Original Implementation)")
        print(f"{'='*80}")
        
        # Create comparison table
        print(f"{'Configuration':<20} {'Val Loss':<10} {'Val Acc':<10} {'Val Perp':<10} {'Time (min)':<12} {'DeepSeek':<10}")
        print(f"{'-'*80}")
        
        for name, result in results.items():
            if "error" not in result:
                val_loss = result.get('val_loss', 0)
                val_acc = result.get('val_accuracy', 0)
                val_perp = result.get('val_perplexity', 0)
                time_min = result.get('training_time_minutes', 0)
                uses_deepseek = "✅" if result.get('uses_deepseek', False) else "❌"
                
                print(f"{name:<20} {val_loss:<10.4f} {val_acc:<10.4f} {val_perp:<10.2f} {time_min:<12.1f} {uses_deepseek:<10}")
            else:
                print(f"{name:<20} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10}")
        
        print(f"{'-'*80}")
        
        # Find best configuration
        valid_results = {name: result for name, result in results.items() 
                        if "error" not in result}
        
        if valid_results:
            best_loss = min(valid_results.items(), key=lambda x: x[1].get('val_loss', float('inf')))
            best_acc = max(valid_results.items(), key=lambda x: x[1].get('val_accuracy', 0))
            best_perp = min(valid_results.items(), key=lambda x: x[1].get('val_perplexity', float('inf')))
            
            print(f"\n🏆 Best Results:")
            print(f"   Lowest Loss: {best_loss[0]} ({best_loss[1]['val_loss']:.4f})")
            print(f"   Highest Accuracy: {best_acc[0]} ({best_acc[1]['val_accuracy']:.4f})")
            print(f"   Lowest Perplexity: {best_perp[0]} ({best_perp[1]['val_perplexity']:.2f})")
        
        print(f"{'='*80}")


def main():
    """Main function to run Experiment 1 with original DeepSeek components"""
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create base configuration
    base_config = MoEModelConfig()
    
    # Create trainer
    trainer = Experiment1ImportTrainer(base_config)
    
    # Run experiment
    results = trainer.run_experiment()
    
    print(f"\n✅ Experiment 1 completed!")
    print(f"📁 Results saved in: {trainer.output_dir}")


if __name__ == "__main__":
    main()
