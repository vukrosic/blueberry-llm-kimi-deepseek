# Experiment 6: DeepseekV3FlashAttention2 vs Baseline Attention Comparison

## Overview
This experiment compares the performance of DeepseekV3FlashAttention2 (from deepseek_modeling.py) against your baseline attention implementation in a transformer architecture.

## Models Compared

### Baseline Attention Model
- Uses your existing MultiHeadAttention implementation
- Standard PyTorch scaled_dot_product_attention
- Simple QKV projection and output projection
- Standard RoPE implementation

### DeepseekV3FlashAttention2 Model  
- Uses DeepseekV3FlashAttention2 from deepseek_modeling.py
- Flash Attention 2 for memory-efficient attention computation
- Advanced QKV projection with LoRA adapters
- Memory-efficient attention with better scaling

## Key Differences

The main difference is in the attention implementation:

**Baseline Attention:**
```python
# Your existing implementation
class MultiHeadAttention(nn.Module):
    # Standard QKV projection
    # scaled_dot_product_attention
    # Standard RoPE
```

**DeepseekV3FlashAttention2:**
```python
# DeepSeek's implementation
class DeepseekV3FlashAttention2(DeepseekV3Attention):
    # Flash Attention 2
    # Advanced QKV projection with LoRA
    # Memory-efficient attention computation
```

## Fair Comparison Setup

Both models use:
- **Same architecture**: 256d, 3L, 4H
- **Same MoE configuration**: 8 experts, top-2 selection (using your implementation)
- **Same training setup**: 1000 steps, batch size 16
- **Same dataset**: 100,000 tokens
- **Same MoE implementation**: Your existing MixtureOfExperts
- **Same normalization**: Standard RMSNorm

## Files

- `exp6_baseline_attention_model.py` - Baseline model with your attention implementation
- `exp6_flash_attention_model.py` - Model using DeepseekV3FlashAttention2
- `exp6_trainer.py` - Training script that compares both models
- `README.md` - This documentation

## Running the Experiment

```bash
cd /root/blueberry-llm-kimi-deepseek/experiments/exp6
python exp6_trainer.py
```

## Expected Results

The experiment will:
1. Train both models for 1000 steps
2. Evaluate performance on validation set
3. Compare validation loss, accuracy, and perplexity
4. Generate loss vs time plots
5. Save results to `exp6_results/exp6_attention_comparison.json`

## Configuration

- **Training Steps**: 1000
- **Batch Size**: 64 (optimized for RTX 4090 24GB)
- **Model Size**: 256d, 3L, 4H
- **MoE**: 8 experts, top-2 selection
- **Sequence Length**: 256
- **Evaluation**: Every 100 steps

## Import Strategy

This experiment follows the "import as much as possible" approach:
- Imports `DeepseekV3FlashAttention2` directly from `deepseek_modeling.py`
- Uses existing MoE infrastructure from `models.components`
- Reuses training utilities from `training.trainer`
- Leverages data loading from `data.loader`

## Expected Training Time

- **Per Model**: ~3-5 minutes (faster with larger batch size)
- **Total Experiment**: ~6-10 minutes

The experiment is designed to be fair and unbiased, comparing only the attention implementations while keeping everything else identical, including using your existing MoE implementation for both models.
