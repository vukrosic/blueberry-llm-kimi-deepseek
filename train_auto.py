#!/usr/bin/env python3
"""
Auto-configured training for Blueberry LLM
Just run: python train_auto.py
"""

import torch
from torch.utils.data import DataLoader, random_split
from auto_config import auto_configure
from llm import train_moe_model, load_and_cache_data, TextTokenDataset

def main():
    print("🫐 Starting Blueberry LLM Auto-Training")
    
    # Auto-configure everything
    configurator = auto_configure()
    configurator.print_config()
    
    # Get model configuration
    model_config = configurator.get_model_config()
    
    # Auto-size dataset based on hardware
    if configurator.config.num_gpus == 0:
        model_config.num_documents = 500
        model_config.max_tokens = 50000
    elif configurator.config.gpu_memory_gb < 16:
        model_config.num_documents = 1000
        model_config.max_tokens = 100000
    elif configurator.config.num_gpus <= 2:
        model_config.num_documents = 2000
        model_config.max_tokens = 250000
    else:
        model_config.num_documents = 5000
        model_config.max_tokens = 500000
    
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
    
    # Create data loaders
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
    model, final_metrics = train_moe_model(model_config, train_loader, val_loader)
    
    # Save results
    print("\n💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'auto_config': configurator.config,
        'final_metrics': final_metrics
    }, 'blueberry_model.pt')
    
    print("✅ Training complete!")
    print(f"   Final validation loss: {final_metrics['val_loss']:.4f}")
    print(f"   Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Model saved as: blueberry_model.pt")

if __name__ == "__main__":
    main()
