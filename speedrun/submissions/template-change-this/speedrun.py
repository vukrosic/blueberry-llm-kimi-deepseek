#!/usr/bin/env python3
"""
T4 Speedrun Challenge Entry Point

This script runs the T4 speedrun challenge with strict timing and validation.
Participants can modify this script and the config to optimize for the lowest validation loss.

Usage:
    python speedrun/speedrun.py                    # Use default config
    python speedrun/speedrun.py --config custom   # Use custom config
    python speedrun/speedrun.py --time-limit 20   # Custom time limit
"""

import os
import sys
import time
import json
import argparse
import torch
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import our modules
from speedrun.config import (
    T4SpeedrunConfig, 
    get_t4_speedrun_config,
    create_custom_t4_config,
    get_memory_optimized_config,
    get_performance_optimized_config,
    get_balanced_config
)
from data import load_and_cache_data, TextTokenDataset
from models import create_model
from training import train_model, validate_training_setup
from training.evaluation import evaluate_model
from system import print_system_info


class SpeedrunTimer:
    """Timer for speedrun challenge."""
    
    def __init__(self, time_limit_minutes: int):
        self.time_limit_seconds = time_limit_minutes * 60
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start the speedrun timer."""
        self.start_time = time.time()
        print(f"⏱️ Speedrun started! Time limit: {self.time_limit_seconds/60:.1f} minutes")
        
    def check_time_limit(self) -> bool:
        """Check if time limit has been exceeded."""
        if self.start_time is None:
            return True
        
        elapsed = time.time() - self.start_time
        remaining = self.time_limit_seconds - elapsed
        
        if remaining <= 0:
            print(f"⏰ TIME'S UP! Speedrun exceeded {self.time_limit_seconds/60:.1f} minute limit")
            return False
        
        return True
    
    def get_remaining_time(self) -> float:
        """Get remaining time in seconds."""
        if self.start_time is None:
            return self.time_limit_seconds
        
        elapsed = time.time() - self.start_time
        return max(0, self.time_limit_seconds - elapsed)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        return time.time() - self.start_time
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time if self.start_time else 0
        print(f"⏱️ Speedrun completed in {elapsed/60:.2f} minutes")


class SpeedrunValidator:
    """Validator for speedrun constraints."""
    
    def __init__(self, config: T4SpeedrunConfig):
        self.config = config
        
    def validate_hardware(self) -> bool:
        """Validate hardware requirements."""
        print("🔍 Validating hardware requirements...")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available - T4 speedrun requires GPU")
            return False
        
        # Check GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 14:  # Allow some tolerance
            print(f"❌ Insufficient GPU memory: {gpu_memory_gb:.1f}GB < 14GB required")
            return False
        
        # Check GPU name (should be T4 or similar)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU: {gpu_name} ({gpu_memory_gb:.1f}GB)")
        
        return True
    
    def validate_config(self) -> bool:
        """Validate configuration constraints."""
        print("🔍 Validating speedrun configuration...")
        return self.config.validate_speedrun_constraints()
    
    def validate_setup(self, model, train_loader, val_loader) -> bool:
        """Validate training setup."""
        print("🔍 Validating training setup...")
        return validate_training_setup(model, train_loader, val_loader, self.config)


class SpeedrunResults:
    """Container for speedrun results."""
    
    def __init__(self, config: T4SpeedrunConfig, timer: SpeedrunTimer):
        self.config = config
        self.timer = timer
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': config.get_speedrun_info(),
            'final_val_loss': float('inf'),
            'best_val_loss': float('inf'),
            'final_step': 0,
            'total_time_minutes': 0,
            'steps_per_second': 0,
            'memory_usage_gb': 0,
            'completed': False,
            'time_exceeded': False,
            'error': None
        }
    
    def update_final_results(self, model, val_loader, final_step: int):
        """Update results with final evaluation."""
        try:
            # Final evaluation
            eval_metrics = evaluate_model(model, val_loader, self.config)
            
            self.results.update({
                'final_val_loss': eval_metrics['val_loss'],
                'final_val_accuracy': eval_metrics['val_accuracy'],
                'final_val_perplexity': eval_metrics['val_perplexity'],
                'final_step': final_step,
                'total_time_minutes': self.timer.get_elapsed_time() / 60,
                'steps_per_second': final_step / self.timer.get_elapsed_time() if self.timer.get_elapsed_time() > 0 else 0,
                'completed': True,
                'time_exceeded': not self.timer.check_time_limit()
            })
            
            # Estimate memory usage
            self.results['memory_usage_gb'] = self.config.estimate_memory_usage()
            
        except Exception as e:
            self.results['error'] = str(e)
            self.results['completed'] = False
    
    def save_results(self, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"speedrun_results_{timestamp}.json"
        
        results_path = Path(current_dir) / "results" / filename
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to {results_path}")
        return results_path
    
    def print_summary(self):
        """Print speedrun summary."""
        print("\n" + "="*60)
        print("🏁 T4 SPEEDRUN RESULTS")
        print("="*60)
        
        if self.results['completed']:
            print(f"✅ Status: COMPLETED")
            print(f"🎯 Final Validation Loss: {self.results['final_val_loss']:.6f}")
            print(f"📊 Final Accuracy: {self.results['final_val_accuracy']:.4f}")
            print(f"📈 Final Perplexity: {self.results['final_val_perplexity']:.2f}")
            print(f"⏱️ Total Time: {self.results['total_time_minutes']:.2f} minutes")
            print(f"🚀 Steps/Second: {self.results['steps_per_second']:.2f}")
            print(f"💾 Memory Usage: {self.results['memory_usage_gb']:.2f} GB")
            
            if self.results['time_exceeded']:
                print("⚠️ WARNING: Exceeded time limit!")
        else:
            print(f"❌ Status: FAILED")
            if self.results['error']:
                print(f"💥 Error: {self.results['error']}")
        
        print("="*60)


def setup_speedrun_data(config: T4SpeedrunConfig):
    """Setup data with fixed seed for reproducibility."""
    print("\n📚 Setting up speedrun data...")
    
    # Use fixed seed for reproducible dataset
    torch.manual_seed(config.SPEEDRUN_DATASET_SEED)
    
    # Load and cache data
    texts, tokenizer, tokens = load_and_cache_data(config)
    
    # Create dataset
    dataset = TextTokenDataset(tokens, config.max_seq_len)
    
    # Fixed train/validation split
    val_size = int(len(dataset) * config.SPEEDRUN_VAL_SPLIT)
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(config.SPEEDRUN_DATASET_SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✅ Dataset: {len(train_dataset):,} train, {len(val_dataset):,} val samples")
    print(f"✅ Vocab size: {config.vocab_size:,}")
    
    return train_loader, val_loader, tokenizer


def run_speedrun(config: T4SpeedrunConfig, time_limit_minutes: int = None):
    """Run the T4 speedrun challenge."""
    
    # Override time limit if specified
    if time_limit_minutes is not None:
        config.SPEEDRUN_TIME_LIMIT_MINUTES = time_limit_minutes
    
    # Initialize timer and validator
    timer = SpeedrunTimer(config.SPEEDRUN_TIME_LIMIT_MINUTES)
    validator = SpeedrunValidator(config)
    results = SpeedrunResults(config, timer)
    
    print("🚀 T4 SPEEDRUN CHALLENGE")
    print("="*60)
    print(f"🎯 Objective: Lowest validation loss in {config.SPEEDRUN_TIME_LIMIT_MINUTES} minutes")
    print(f"💻 Hardware: Tesla T4 (16GB VRAM)")
    print(f"📊 Dataset: {config.max_tokens:,} tokens, {config.num_documents} documents")
    print("="*60)
    
    try:
        # Validate hardware
        if not validator.validate_hardware():
            raise RuntimeError("Hardware validation failed")
        
        # Validate configuration
        if not validator.validate_config():
            raise RuntimeError("Configuration validation failed")
        
        # Setup data
        train_loader, val_loader, tokenizer = setup_speedrun_data(config)
        
        # Setup model
        print(f"\n🤖 Setting up model...")
        model = create_model(config, "moe")
        model = model.cuda()
        
        # Validate setup
        if not validator.validate_setup(model, train_loader, val_loader):
            raise RuntimeError("Training setup validation failed")
        
        # Start timer
        timer.start()
        
        # Custom training loop with time limit checking
        print(f"\n🏃 Starting speedrun training...")
        model, final_metrics = train_model_with_time_limit(
            model, train_loader, val_loader, config, timer
        )
        
        # Stop timer
        timer.stop()
        
        # Update results
        results.update_final_results(model, val_loader, final_metrics.get('final_step', 0))
        
        # Print summary
        results.print_summary()
        
        # Save results
        results.save_results()
        
        return results
        
    except Exception as e:
        print(f"\n❌ Speedrun failed: {e}")
        results.results['error'] = str(e)
        results.results['completed'] = False
        results.print_summary()
        return results


def train_model_with_time_limit(model, train_loader, val_loader, config, timer):
    """Custom training loop with time limit checking."""
    from training.trainer import train_model_native, TrainingState
    from optimizers import setup_optimizers, get_lr_scheduler
    from torch.amp import GradScaler
    
    # Setup device
    device = torch.device('cuda')
    model = model.to(device)
    
    # Setup optimizers and schedulers
    optimizers = setup_optimizers(model, config, use_warmup=True)
    schedulers = [get_lr_scheduler(opt, config, "cosine_warmup") for opt in optimizers]
    
    # Setup gradient scaler for mixed precision
    scaler = GradScaler('cuda') if config.use_amp else None
    
    # Training state
    training_state = TrainingState(
        model=model,
        optimizers=optimizers,
        schedulers=schedulers,
        scaler=scaler,
        config=config,
        device=device,
        start_step=0
    )
    
    # Modified training loop with time limit checking
    training_state.model.train()
    
    total_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0
    
    while training_state.step < config.max_steps:
        # Check time limit
        if not timer.check_time_limit():
            print(f"\n⏰ Time limit reached at step {training_state.step}")
            break
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if training_state.step >= config.max_steps:
                break
            
            # Check time limit before each batch
            if not timer.check_time_limit():
                print(f"\n⏰ Time limit reached at step {training_state.step}")
                break
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass and backward pass (simplified)
            if config.use_amp:
                with torch.amp.autocast('cuda'):
                    logits, aux_loss = model(x, return_aux_loss=True)
                    ce_loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, config.vocab_size), 
                        y.view(-1)
                    )
                    total_loss_val = ce_loss + (aux_loss if aux_loss is not None else 0)
                    loss = total_loss_val / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                logits, aux_loss = model(x, return_aux_loss=True)
                ce_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, config.vocab_size), 
                    y.view(-1)
                )
                total_loss_val = ce_loss + (aux_loss if aux_loss is not None else 0)
                loss = total_loss_val / config.gradient_accumulation_steps
                loss.backward()
            
            # Accumulate metrics
            total_loss += ce_loss.item()
            total_aux_loss += aux_loss.item() if aux_loss is not None else 0.0
            num_batches += 1
            
            # Increment step counter
            training_state.step += 1
            
            # Optimizer step
            if training_state.step % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                
                # Update learning rate
                for scheduler in schedulers:
                    scheduler.step()
            
            # Logging
            if training_state.step % 100 == 0:
                remaining_time = timer.get_remaining_time()
                print(f"Step {training_state.step}: Loss={ce_loss.item():.4f}, Time remaining: {remaining_time/60:.1f}min")
            
            # Evaluation
            if training_state.step % config.eval_every == 0 and training_state.step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                remaining_time = timer.get_remaining_time()
                print(f"Step {training_state.step}: Val Loss={eval_metrics['val_loss']:.4f}, Time remaining: {remaining_time/60:.1f}min")
                
                # Check for best model
                if eval_metrics['val_loss'] < training_state.best_val_loss:
                    training_state.best_val_loss = eval_metrics['val_loss']
                    print(f"🎉 New best validation loss: {training_state.best_val_loss:.4f}")
    
    # Final evaluation
    final_metrics = evaluate_model(model, val_loader, config)
    final_metrics['final_step'] = training_state.step
    final_metrics['best_val_loss'] = training_state.best_val_loss
    
    return model, final_metrics


def main():
    """Main entry point for speedrun challenge."""
    parser = argparse.ArgumentParser(description="T4 Speedrun Challenge")
    parser.add_argument("--config", type=str, default="default", 
                       choices=["default", "memory", "performance", "balanced"],
                       help="Configuration preset to use")
    parser.add_argument("--time-limit", type=int, default=5,
                       help="Time limit in minutes")
    parser.add_argument("--custom-config", action="store_true",
                       help="Use custom configuration parameters")
    
    # Custom config parameters
    parser.add_argument("--d-model", type=int, help="Model dimension")
    parser.add_argument("--n-layers", type=int, help="Number of layers")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    
    args = parser.parse_args()
    
    # Get configuration
    if args.custom_config:
        config_kwargs = {}
        if args.d_model: config_kwargs['d_model'] = args.d_model
        if args.n_layers: config_kwargs['n_layers'] = args.n_layers
        if args.n_heads: config_kwargs['n_heads'] = args.n_heads
        if args.batch_size: config_kwargs['batch_size'] = args.batch_size
        if args.lr: config_kwargs['muon_lr'] = args.lr
        if args.max_steps: config_kwargs['max_steps'] = args.max_steps
        
        config = create_custom_t4_config(**config_kwargs)
    else:
        config_map = {
            "default": get_t4_speedrun_config,
            "memory": get_memory_optimized_config,
            "performance": get_performance_optimized_config,
            "balanced": get_balanced_config,
        }
        config = config_map[args.config]()
    
    # Print system info
    print_system_info()
    
    # Run speedrun
    results = run_speedrun(config, args.time_limit)
    
    # Exit with appropriate code
    if results.results['completed']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
