#!/usr/bin/env python3
"""
T4-optimized training for Blueberry LLM
Just run: python train.py
"""

import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

import torch
import argparse
from torch.utils.data import DataLoader, random_split
from core.t4_config import t4_configure
from legacy.llm import train_moe_model, load_and_cache_data, TextTokenDataset

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="T4-optimized training for Blueberry LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Single T4 GPU - no Megatron options needed
    
    return parser.parse_args()

def main():
    print("🫐 Starting Blueberry LLM T4-Training")
    
    # Parse arguments
    args = parse_arguments()
    
    # T4-configure everything
    configurator = t4_configure()
    
    # Single T4 GPU - no Megatron support
    print("🚀 Single T4 GPU training - Megatron disabled")
    
    configurator.print_config()
    
    # Print detailed GPU system information
    print("\n🔍 Detailed GPU System Information:")
    print("=" * 50)
    from system import print_system_info
    print_system_info()
    print("=" * 50)
    
    # Single T4 GPU - no distributed training needed
    print("🚀 Single T4 GPU training mode")
    
    # Get model configuration
    model_config = configurator.get_model_config()
    
    # Auto-size dataset for T4 GPU
    model_config.num_documents = 2000
    model_config.max_tokens = 200000
    
    print(f"\n📊 Loading {model_config.num_documents} documents, {model_config.max_tokens:,} tokens...")
    
    # Load data
    texts, tokenizer, tokens = load_and_cache_data(model_config)
    dataset = TextTokenDataset(tokens, model_config.max_seq_len)
    
    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders for single T4 GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=model_config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"   Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Train the model
    print("\n🚀 Starting training...")
    
    # Use Megatron-enabled training if requested
    if configurator.config.use_megatron:
        print("🚀 Using Megatron-enabled training pipeline...")
        from models import create_model
        from training import train_model
        from configs import T4MoEModelConfig
        
        # Convert legacy config to new config format
        t4_config = T4MoEModelConfig(
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            d_ff=model_config.d_ff,
            batch_size=model_config.batch_size,
            max_steps=model_config.max_steps,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps,
            muon_lr=model_config.muon_lr,
            max_seq_len=model_config.max_seq_len,
            num_experts=model_config.num_experts,
            use_amp=model_config.use_amp,
            vocab_size=model_config.vocab_size  # Add vocab_size from tokenizer
        )
        
        # Create model optimized for T4
        model = create_model(t4_config, "moe")
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Train with new pipeline
        model, final_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=t4_config,
            device=device
        )
    else:
        # Use legacy training pipeline
        model, final_metrics = train_moe_model(model_config, train_loader, val_loader)
    
    # Save results
    print("\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        't4_config': configurator.config,
        'tokenizer': tokenizer,
        'final_metrics': final_metrics
    }, 'blueberry_model.pt')
    
    print("✅ Training complete!")
    print(f"   Final validation loss: {final_metrics['val_loss']:.4f}")
    print(f"   Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Model saved as: blueberry_model.pt")

if __name__ == "__main__":
    main()
