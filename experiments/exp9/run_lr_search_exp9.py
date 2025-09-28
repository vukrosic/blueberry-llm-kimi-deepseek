#!/usr/bin/env python3
"""
Learning Rate Search Runner for Experiment 9
Runs systematic learning rate search and integrates with existing experiment 9
"""

import torch
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from experiments.exp9.lr_search_exp9 import LearningRateSearchTrainer
from experiments.exp9.exp9_trainer import LongTermExperiment9Trainer


def run_learning_rate_search():
    """Run comprehensive learning rate search"""
    print("🚀 Learning Rate Search for Experiment 9")
    print("=" * 60)
    
    # Create configuration for learning rate search
    base_config = MoEModelConfig(
        max_steps=2000,  # Shorter training for LR search
        batch_size=128,
        max_tokens=100000,
        eval_every=100,
        num_documents=1000,
        max_seq_len=256,
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        num_experts=8,
        expert_top_k=2,
    )
    
    print(f"📋 Configuration:")
    print(f"   Model: Attention+MLP 512d")
    print(f"   Batch Size: {base_config.batch_size}")
    print(f"   Model: {base_config.d_model}d, {base_config.n_layers}L, {base_config.n_heads}H")
    print(f"   Max Steps per LR: {base_config.max_steps}")
    
    # Define learning rates to test (comprehensive range)
    learning_rates = [
        1e-5,   # Very low
        3e-5,   # Low
        1e-4,   # Medium-low
        3e-4,   # Medium
        1e-3,   # Medium-high
        3e-3,   # High
        1e-2,   # Very high
    ]
    
    print(f"📋 Learning rates to test: {learning_rates}")
    
    # Create LR search trainer
    lr_trainer = LearningRateSearchTrainer(base_config, output_dir="exp9_lr_search")
    
    # Run learning rate search
    print(f"\n🧪 Running learning rate search...")
    start_time = time.time()
    
    lr_results = lr_trainer.run_lr_search(
        learning_rates=learning_rates,
        max_steps=2000,              # 2k steps per LR
        eval_every=100,              # Evaluate every 100 steps
        early_stopping_patience=5    # Early stop after 5 evaluations without improvement
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
        
        return best_lr, lr_results
    else:
        print("❌ No valid learning rate results found!")
        return None, lr_results


def run_long_term_training_with_best_lr(best_lr: float):
    """Run long-term training with the best learning rate found"""
    print(f"\n🚀 Long-term Training with Best LR: {best_lr:.2e}")
    print("=" * 60)
    
    # Create configuration for long-term training
    base_config = MoEModelConfig(
        max_steps=10000,  # Long-term training
        batch_size=128,
        max_tokens=100000,
        eval_every=100,
        num_documents=1000,
        max_seq_len=256,
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048,
        num_experts=8,
        expert_top_k=2,
    )
    
    # Create long-term trainer
    long_term_trainer = LongTermExperiment9Trainer(base_config, output_dir="exp9_results_best_lr")
    
    # Modify the trainer to use the best learning rate
    original_train_model = long_term_trainer.train_model_long_term
    
    def train_with_best_lr(model, model_name: str, total_steps: int = 10000, 
                          checkpoint_every: int = 3000, eval_every: int = 100, 
                          hellaswag_every: int = 1000) -> Dict[str, Any]:
        """Modified training function with best learning rate"""
        print(f"\n{'='*80}")
        print(f"🚀 Long-term Training: {model_name} ({total_steps} steps)")
        print(f"🎯 Using Best LR: {best_lr:.2e}")
        print(f"📁 Checkpoints every {checkpoint_every} steps")
        print(f"📊 Evaluation every {eval_every} steps")
        print(f"🧪 HellaSwag benchmark every {hellaswag_every} steps")
        print(f"{'='*80}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup optimizer
        optimizers = setup_muon_optimizer(model, long_term_trainer.base_config)
        
        # Override learning rates with best LR
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = best_lr
        
        # Learning rate schedule with longer warmup
        schedulers = []
        for optimizer in optimizers:
            warmup_steps = max(100, total_steps // 20)  # 5% warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            schedulers.append(scheduler)
        
        # Data loaders
        train_loader = DataLoader(
            long_term_trainer.train_dataset, 
            batch_size=long_term_trainer.base_config.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        val_loader = DataLoader(
            long_term_trainer.val_dataset, 
            batch_size=long_term_trainer.base_config.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        # Training loop (rest of the function remains the same as original)
        model.train()
        step = 0
        start_time = time.time()
        
        # Track loss curves
        eval_steps = []
        eval_losses = []
        eval_times = []
        train_losses = []
        
        print(f"🚀 Starting long-term training for {total_steps} steps...")
        
        while step < total_steps:
            for batch_idx, (x, y) in enumerate(train_loader):
                if step >= total_steps:
                    break
                
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                logits, aux_loss = model(x, return_aux_loss=True)
                ce_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, long_term_trainer.vocab_size), y.view(-1)
                )
                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss
                loss = total_loss / long_term_trainer.base_config.gradient_accumulation_steps
                loss.backward()
                
                # Optimizer step
                if (step + 1) % long_term_trainer.base_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), long_term_trainer.base_config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                
                # Progress logging (every eval_every steps)
                if step % eval_every == 0:
                    print(f"Step {step}/{total_steps}: Loss={ce_loss.item():.6f}")
                    train_losses.append(ce_loss.item())
                
                # Evaluation (every eval_every steps)
                if step % eval_every == 0 and step > 0:
                    eval_metrics = long_term_trainer._evaluate_model(model, val_loader)
                    
                    # Track for plotting
                    eval_steps.append(step)
                    eval_losses.append(eval_metrics['val_loss'])
                    eval_times.append(time.time())
                    
                    print(f"   Val Loss: {eval_metrics['val_loss']:.6f}, Val Acc: {eval_metrics['val_accuracy']:.4f}")
                
                # Checkpoint saving (every checkpoint_every steps)
                if step % checkpoint_every == 0 and step > 0:
                    long_term_trainer._save_checkpoint(model, step, eval_losses[-1] if eval_losses else 0)
                
                # HellaSwag benchmark evaluation (every hellaswag_every steps)
                if step % hellaswag_every == 0 and step > 0:
                    print(f"\n🧪 Running HellaSwag benchmark at step {step}...")
                    try:
                        benchmark_result = long_term_trainer.hellaswag_evaluator.evaluate_model(model, model_name)
                        print(f"✅ HellaSwag benchmark completed: {benchmark_result}")
                    except Exception as e:
                        print(f"❌ HellaSwag benchmark failed: {e}")
                
                step += 1
        
        total_time = time.time() - start_time
        
        # Final evaluation
        print(f"\n🔍 Final evaluation...")
        final_eval = long_term_trainer._evaluate_model(model, val_loader)
        
        # Add final evaluation to tracking
        eval_steps.append(total_steps)
        eval_losses.append(final_eval['val_loss'])
        eval_times.append(time.time())
        
        # Store loss curve data
        long_term_trainer.loss_curves[model_name] = {
            'eval_steps': eval_steps,
            'eval_losses': eval_losses,
            'eval_times': eval_times,
            'train_losses': train_losses
        }
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            **final_eval,
            'model_name': model_name,
            'learning_rate': best_lr,
            'training_time_minutes': total_time / 60,
            'parameter_count': param_count,
            'parameters_millions': param_count / 1e6,
            'total_steps': total_steps,
            'checkpoint_every': checkpoint_every,
            'eval_every': eval_every
        }
        
        print(f"✅ {model_name} Long-term Training Results:")
        print(f"   Val Loss: {results['val_loss']:.6f}")
        print(f"   Val Acc: {results['val_accuracy']:.6f}")
        print(f"   Val Perp: {results['val_perplexity']:.4f}")
        print(f"   Time: {results['training_time_minutes']:.2f} min")
        print(f"   Params: {results['parameters_millions']:.2f}M")
        print(f"   Steps: {results['total_steps']}")
        print(f"   LR: {results['learning_rate']:.2e}")
        
        return results
    
    # Replace the training method
    long_term_trainer.train_model_long_term = train_with_best_lr
    
    # Run long-term training with best LR
    print(f"\n🧪 Running long-term training with best LR...")
    
    results = long_term_trainer.run_long_term_training(
        total_steps=10000,      # Train for 10k steps
        checkpoint_every=3000,   # Save checkpoint every 3k steps
        eval_every=100,         # Evaluate every 100 steps
        hellaswag_every=1000    # HellaSwag benchmark every 1k steps
    )
    
    print(f"\n✅ Long-term Training with Best LR completed!")
    print(f"📁 Results saved in: {long_term_trainer.output_dir}")
    
    return results


def main():
    """Main function to run complete learning rate search and training"""
    print("🚀 Complete Learning Rate Search and Training for Experiment 9")
    print("=" * 80)
    
    try:
        # Step 1: Run learning rate search
        print("\n" + "="*60)
        print("STEP 1: Learning Rate Search")
        print("="*60)
        
        best_lr, lr_results = run_learning_rate_search()
        
        if best_lr is None:
            print("❌ Learning rate search failed!")
            return False
        
        # Step 2: Run long-term training with best LR
        print("\n" + "="*60)
        print("STEP 2: Long-term Training with Best LR")
        print("="*60)
        
        long_term_results = run_long_term_training_with_best_lr(best_lr)
        
        # Step 3: Summary
        print("\n" + "="*60)
        print("STEP 3: Summary")
        print("="*60)
        
        print(f"✅ Complete Experiment 9 Learning Rate Search and Training:")
        print(f"   🏆 Best Learning Rate: {best_lr:.2e}")
        print(f"   📊 LR Search Results: exp9_lr_search/")
        print(f"   🚀 Long-term Training Results: exp9_results_best_lr/")
        print(f"   📈 Final Val Loss: {long_term_results.get('val_loss', 'N/A')}")
        print(f"   📈 Final Val Acc: {long_term_results.get('val_accuracy', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in complete experiment: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
