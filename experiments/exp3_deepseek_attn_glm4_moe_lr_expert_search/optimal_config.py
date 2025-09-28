"""
Optimal Configuration from Experiment 3
DeepSeek Attention + GLM4 MoE with optimized hyperparameters
"""

from configs.moe_config import MoEModelConfig

def get_optimal_exp3_config():
    """
    Returns the optimal configuration from Experiment 3
    
    Results:
    - Final Validation Loss: 0.0614
    - Final Validation Accuracy: 98.73%
    - Final Validation Perplexity: 1.0634
    - Training Time: 26.6 minutes
    - Optimal Learning Rate: 3e-3
    """
    return MoEModelConfig(
        # Model Architecture (optimized for memory efficiency)
        d_model=256,           # Hidden size
        n_heads=8,             # Attention heads
        n_layers=6,            # Transformer layers
        d_ff=512,              # Feed-forward dimension (MoE-optimized)
        
        # MoE Configuration (optimal from experiment)
        num_experts=4,         # Number of experts
        expert_top_k=2,       # Top-k routing
        
        # Training Configuration (optimal learning rate)
        muon_lr=3e-3,         # Optimal learning rate from LR search
        batch_size=16,        # Batch size for extended training
        max_seq_len=128,      # Sequence length
        
        # Data Configuration (memory optimized)
        max_tokens=50000,     # Token limit for memory efficiency
        num_documents=1000,   # Number of documents
        eval_every=100,       # Evaluation frequency
        
        # Training Schedule (proven effective)
        max_steps=10000,      # Total training steps
        checkpoint_every=3000, # Checkpoint frequency
        
        # Additional settings
        dropout=0.1,          # Dropout rate
        weight_decay=0.01,    # Weight decay
        use_amp=False,        # Mixed precision (can enable if needed)
    )

def get_optimal_exp3_config_large():
    """
    Returns a larger version of the optimal configuration
    Use this if you have more GPU memory available
    """
    config = get_optimal_exp3_config()
    
    # Scale up model size
    config.d_model = 512
    config.n_layers = 12
    config.d_ff = 1024
    config.num_experts = 8
    config.max_seq_len = 256
    config.batch_size = 8  # Reduce batch size for larger model
    
    return config

def get_optimal_exp3_config_small():
    """
    Returns a smaller version of the optimal configuration
    Use this for very limited GPU memory
    """
    config = get_optimal_exp3_config()
    
    # Scale down model size
    config.d_model = 128
    config.n_layers = 4
    config.d_ff = 256
    config.num_experts = 2
    config.max_seq_len = 64
    config.batch_size = 32  # Increase batch size for smaller model
    
    return config

# Export the main configuration
OPTIMAL_EXP3_CONFIG = get_optimal_exp3_config()

# Configuration variants
OPTIMAL_EXP3_CONFIG_LARGE = get_optimal_exp3_config_large()
OPTIMAL_EXP3_CONFIG_SMALL = get_optimal_exp3_config_small()

# Configuration metadata
EXP3_RESULTS = {
    'final_val_loss': 0.0614,
    'final_val_accuracy': 0.9873,
    'final_val_perplexity': 1.0634,
    'training_time_minutes': 26.6,
    'optimal_learning_rate': 3e-3,
    'total_steps': 10000,
    'model_parameters': 20500000,  # ~20.5M parameters
    'architecture': 'DeepSeek Attention + GLM4 MoE',
    'experiment': 'exp3_deepseek_attn_glm4_moe_lr_expert_search'
}
