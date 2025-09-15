#!/usr/bin/env python3
"""
Training benchmark comparing Megatron vs Native implementation.
Tests both implementations with 500, 1000, and 2000 steps.
"""

import time
import torch
from configs import AdaptiveMoEModelConfig
from models import create_model
from models.megatron_wrapper import is_megatron_model
from data import load_and_cache_data, TextTokenDataset
from training import train_model
from torch.utils.data import DataLoader, random_split

def benchmark_training(use_megatron, config, steps, model_type="moe"):
    """Benchmark training for a specific number of steps."""
    print(f"\n{'='*60}")
    backend_name = "Megatron" if use_megatron else "Native"
    print(f"🚀 Training {backend_name} Implementation - {steps} steps")
    print(f"{'='*60}")
    
    # Setup configuration
    config.use_megatron = use_megatron
    config.max_steps = steps
    config.vocab_size = 49152
    
    # Setup data
    print("📚 Setting up data...")
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    # Train/validation split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    # Create model
    print(f"🤖 Creating {backend_name} model...")
    model = create_model(config, model_type)
    
    # Check if it's actually using Megatron
    if use_megatron:
        is_megatron = is_megatron_model(model)
        print(f"   Is Megatron wrapped: {is_megatron}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train
    print(f"🎯 Starting training for {steps} steps...")
    start_time = time.time()
    
    try:
        trained_model, final_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        training_time = time.time() - start_time
        
        # Results
        print(f"\n📊 {backend_name} Results ({steps} steps):")
        print(f"   Training time: {training_time/60:.2f} minutes")
        print(f"   Final validation loss: {final_metrics['val_loss']:.4f}")
        print(f"   Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"   Steps per second: {final_metrics.get('steps_per_second', 0):.2f}")
        print(f"   Model FLOPs Utilization: {final_metrics.get('mfu', 0)*100:.1f}%")
        
        return {
            'backend': backend_name,
            'steps': steps,
            'training_time_minutes': training_time / 60,
            'val_loss': final_metrics['val_loss'],
            'val_accuracy': final_metrics['val_accuracy'],
            'steps_per_second': final_metrics.get('steps_per_second', 0),
            'mfu': final_metrics.get('mfu', 0),
            'is_megatron': is_megatron_model(trained_model) if use_megatron else False
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def main():
    print("🚀 Blueberry LLM: Training Benchmark (Megatron vs Native)")
    print("=" * 80)
    
    # Configuration
    config = AdaptiveMoEModelConfig()
    config.batch_size = 16  # Reasonable batch size
    config.eval_every = 100  # Evaluate every 100 steps
    config.eval_steps = 50   # Quick evaluation
    
    print(f"📊 Benchmark Configuration:")
    print(f"   Model: {config.d_model}d-{config.n_layers}L-{config.n_heads}H")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Sequence length: {config.max_seq_len}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   Test steps: [500, 1000, 2000]")
    print()
    
    # Test configurations
    test_steps = [500, 1000, 2000]
    results = []
    
    # Test both implementations
    for steps in test_steps:
        # Native implementation
        result_native = benchmark_training(False, config, steps)
        if result_native:
            results.append(result_native)
        
        # Megatron implementation
        result_megatron = benchmark_training(True, config, steps)
        if result_megatron:
            results.append(result_megatron)
    
    # Summary
    print(f"\n{'='*80}")
    print("📈 TRAINING BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    for steps in test_steps:
        print(f"\n🎯 {steps} Steps Comparison:")
        native_results = [r for r in results if r['steps'] == steps and r['backend'] == 'Native']
        megatron_results = [r for r in results if r['steps'] == steps and r['backend'] == 'Megatron']
        
        if native_results and megatron_results:
            native = native_results[0]
            megatron = megatron_results[0]
            
            print(f"   Native:  {native['training_time_minutes']:.2f}min, loss={native['val_loss']:.4f}, {native['steps_per_second']:.2f} steps/s")
            print(f"   Megatron: {megatron['training_time_minutes']:.2f}min, loss={megatron['val_loss']:.4f}, {megatron['steps_per_second']:.2f} steps/s")
            
            # Speed comparison
            if megatron['training_time_minutes'] < native['training_time_minutes']:
                speedup = native['training_time_minutes'] / megatron['training_time_minutes']
                print(f"   🚀 Megatron is {speedup:.2f}x faster!")
            else:
                slowdown = megatron['training_time_minutes'] / native['training_time_minutes']
                print(f"   📋 Native is {slowdown:.2f}x faster")
            
            # Throughput comparison
            if megatron['steps_per_second'] > native['steps_per_second']:
                throughput_gain = megatron['steps_per_second'] / native['steps_per_second']
                print(f"   📊 Megatron throughput: {throughput_gain:.2f}x higher")
            else:
                throughput_loss = native['steps_per_second'] / megatron['steps_per_second']
                print(f"   📊 Native throughput: {throughput_loss:.2f}x higher")
    
    print(f"\n{'='*80}")
    print("✅ Benchmark completed!")

if __name__ == "__main__":
    main()
