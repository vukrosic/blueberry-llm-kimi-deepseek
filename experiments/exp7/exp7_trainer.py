"""
Experiment 7 Trainer: Best Architecture Training
Trains the attention_mlp model that achieved best efficiency in exp6
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from configs.moe_config import MoEModelConfig
from experiments.exp7.exp7_attention_mlp_model import create_exp7_model
from data.loader import load_and_cache_data
from utils.helpers import set_seed
from training.evaluation import evaluate_model


class Exp7Trainer:
    """Trainer for Experiment 7: Best Architecture Model"""
    
    def __init__(self, config: MoEModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.tokenizer = None  # For decoding sequences
        
        # Results tracking
        self.results = {
            "model_name": "exp7_attention_mlp",
            "config": config.__dict__,
            "training_history": [],
            "final_metrics": {},
            "training_time_minutes": 0,
            "parameter_count": 0,
            "test_mode": False,
            "max_steps": config.max_steps
        }
        
        print(f"🔍 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        print(f"🚀 Experiment 7 Configuration:")
        print(f"   Model: Best Architecture (attention_mlp from exp6)")
        print(f"   Steps: {config.max_steps}")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   Model: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
        print(f"   Expected: ~150M params, ~15-20 min training, <1.0 loss")
    
    def load_data(self):
        """Load and prepare data"""
        print("📦 Loading cached data...")
        
        # Load data using the existing loader
        texts, tokenizer, tokens = load_and_cache_data(self.config)
        
        # Create random window split to avoid temporal bias
        import random
        random.seed(42)  # For reproducibility
        
        # Create all possible windows
        all_windows = list(range(max(0, len(tokens) - self.config.max_seq_len)))
        
        # Randomly split windows (not tokens)
        random.shuffle(all_windows)
        split_idx = int(0.9 * len(all_windows))
        train_windows = all_windows[:split_idx]
        val_windows = all_windows[split_idx:]
        
        # Create datasets from window indices
        from data.loader import TextTokenDataset
        train_data = TextTokenDataset(tokens, self.config.max_seq_len, train_windows)
        val_data = TextTokenDataset(tokens, self.config.max_seq_len, val_windows)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_data, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_data, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            drop_last=True
        )
        
        print(f"📊 Dataset: {len(train_data)} train, {len(val_data)} val samples")
        print(f"📊 Total windows: {len(all_windows)}, Train: {len(train_windows)}, Val: {len(val_windows)}")
        
        # Store tokenizer for sequence decoding
        self.tokenizer = tokenizer
        
        return {"texts": texts, "tokenizer": tokenizer, "tokens": tokens}
    
    def create_model(self):
        """Create the exp7 model"""
        print("🏗️ Creating Exp7 model...")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create model
        self.model = create_exp7_model(self.config)
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.results["parameter_count"] = total_params
        self.results["parameters_millions"] = total_params / 1e6
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Parameters (M): {total_params / 1e6:.2f}M")
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,  # Fixed learning rate
            weight_decay=self.config.weight_decay
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps,
            eta_min=1e-4  # Fixed minimum learning rate
        )
        
        print("✅ Model created successfully")
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get input and target from TextTokenDataset
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass
        logits, aux_loss = self.model(input_ids, return_aux_loss=True)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Add auxiliary loss if present
        if aux_loss is not None:
            loss = loss + aux_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def print_sequences(self, input_ids, target_ids, logits, step, mode="train"):
        """Print input, target, and predicted sequences for debugging"""
        if self.tokenizer is None:
            return
            
        # Get the first sequence in the batch
        input_seq = input_ids[0].cpu().tolist()
        target_seq = target_ids[0].cpu().tolist()
        predicted_seq = logits[0].argmax(dim=-1).cpu().tolist()
        
        # Decode sequences
        input_text = self.tokenizer.decode(input_seq, skip_special_tokens=True)
        target_text = self.tokenizer.decode(target_seq, skip_special_tokens=True)
        predicted_text = self.tokenizer.decode(predicted_seq, skip_special_tokens=True)
        
        print(f"\n🔍 {mode.upper()} SEQUENCES at step {step}:")
        print(f"   Input:     '{input_text[:100]}{'...' if len(input_text) > 100 else ''}'")
        print(f"   Target:    '{target_text[:100]}{'...' if len(target_text) > 100 else ''}'")
        print(f"   Predicted: '{predicted_text[:100]}{'...' if len(predicted_text) > 100 else ''}'")
        
        # Show token-level comparison for first 20 tokens
        print(f"   Token comparison (first 20):")
        for i in range(min(20, len(target_seq))):
            target_token = self.tokenizer.decode([target_seq[i]], skip_special_tokens=True)
            pred_token = self.tokenizer.decode([predicted_seq[i]], skip_special_tokens=True)
            match = "✓" if target_seq[i] == predicted_seq[i] else "✗"
            print(f"     {i:2d}: {match} Target='{target_token}' Pred='{pred_token}'")
        print()
    
    def train(self):
        """Train the model"""
        print(f"🚀 Starting training for {self.config.max_steps} steps...")
        
        start_time = time.time()
        step = 0
        
        # Training loop
        while step < self.config.max_steps:
            for batch in self.train_loader:
                if step >= self.config.max_steps:
                    break
                
                # Training step
                loss = self.train_step(batch)
                
                # Log progress
                if step % 100 == 0:
                    print(f"Step {step}/{self.config.max_steps}: Loss={loss:.4f}")
                
                # Print sequences occasionally during training
                if step % 500 == 0 and step > 0:
                    input_ids, target_ids = batch
                    input_ids = input_ids.to(self.device)
                    target_ids = target_ids.to(self.device)
                    with torch.no_grad():
                        output = self.model(input_ids, return_aux_loss=False)
                        logits = output if not isinstance(output, tuple) else output[0]
                    self.print_sequences(input_ids, target_ids, logits, step, "train")
                
                # Evaluation
                if step % self.config.eval_every == 0 and step > 0:
                    eval_results = evaluate_model(self.model, self.val_loader, self.config)
                    val_loss = eval_results['val_loss']
                    val_acc = eval_results['val_accuracy']
                    val_perp = eval_results['val_perplexity']
                    
                    # Record metrics
                    self.results["training_history"].append({
                        "step": step,
                        "train_loss": loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_perplexity": val_perp,
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })
                    
                    print(f"🔍 Evaluating model at step {step}...")
                    print(f"✅ Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                    # Print validation sequences
                    val_batch = next(iter(self.val_loader))
                    val_input_ids, val_target_ids = val_batch
                    val_input_ids = val_input_ids.to(self.device)
                    val_target_ids = val_target_ids.to(self.device)
                    with torch.no_grad():
                        val_output = self.model(val_input_ids, return_aux_loss=False)
                        val_logits = val_output if not isinstance(val_output, tuple) else val_output[0]
                    self.print_sequences(val_input_ids, val_target_ids, val_logits, step, "val")
                
                step += 1
        
        # Final evaluation
        print("🔍 Final evaluation...")
        eval_results = evaluate_model(self.model, self.val_loader, self.config)
        val_loss = eval_results['val_loss']
        val_acc = eval_results['val_accuracy']
        val_perp = eval_results['val_perplexity']
        
        # Record final metrics
        self.results["final_metrics"] = {
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_perplexity": val_perp
        }
        
        # Training time
        training_time = (time.time() - start_time) / 60
        self.results["training_time_minutes"] = training_time
        
        print(f"✅ Exp7 Results:")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val Acc: {val_acc:.4f}")
        print(f"   Val Perp: {val_perp:.2f}")
        print(f"   Time: {training_time:.2f} min")
        print(f"   Params: {self.results['parameters_millions']:.2f}M")
        print("✅ Exp7 training completed")
    
    def save_results(self):
        """Save training results"""
        os.makedirs("experiments/exp7/exp7_results", exist_ok=True)
        
        # Save results
        results_file = "experiments/exp7/exp7_results/exp7_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save model
        model_file = "experiments/exp7/exp7_results/exp7_model.pt"
        torch.save(self.model.state_dict(), model_file)
        
        print(f"💾 Results saved to: {results_file}")
        print(f"💾 Model saved to: {model_file}")
    
    def run_experiment(self):
        """Run the complete experiment"""
        print("=" * 80)
        print("🚀 Starting Experiment 7: Best Architecture Training")
        print("=" * 80)
        
        # Load data
        self.load_data()
        
        # Create model
        self.create_model()
        
        # Train model
        self.train()
        
        # Save results
        self.save_results()
        
        print("=" * 80)
        print("✅ Experiment 7 completed!")
        print("=" * 80)


def main():
    """Main function"""
    # Configuration - Extended training for 1 hour (17,366 steps)
    config = MoEModelConfig(
        max_steps=17366,  # Training for ~1 hour (8.7x more steps)
        batch_size=32,    # Reduced batch size for larger model
        max_tokens=100000,
        eval_every=1000,  # Evaluate every 1000 steps for extended training
        num_documents=1000,
        max_seq_len=256,
        d_model=768,     # Increased from 256 to 768
        n_heads=12,      # Increased from 4 to 12 (768/12 = 64 head dim)
        n_layers=8,      # Increased from 3 to 8 layers
        d_ff=3072,       # Increased from 1024 to 3072 (4x d_model)
        num_experts=8,   # Not used in MLP model
        expert_top_k=2,  # Not used in MLP model
        vocab_size=32000,
        weight_decay=0.01,
        eval_steps=50,
        use_amp=False
    )
    
    # Create trainer and run experiment
    trainer = Exp7Trainer(config)
    trainer.run_experiment()


if __name__ == "__main__":
    main()
