# Experiment 5: DeepseekV3MoE vs Baseline MoE Comparison

## Overview
This experiment compares the performance of DeepseekV3MoE (from deepseek_modeling.py) against your baseline MoE implementation in a transformer architecture.

## Models Compared

### Baseline MoE Model
- Uses your existing MixtureOfExperts implementation
- Standard MLP experts with SiLU activation
- Simple top-k routing with softmax weights
- Basic load balancing loss

### DeepseekV3MoE Model  
- Uses DeepseekV3MoE from deepseek_modeling.py
- DeepseekV3MLP experts (gated architecture)
- Advanced MoEGate with sigmoid scoring
- Sophisticated routing with group-based selection

## Key Differences

The main difference is in the MoE implementation:

**Baseline MoE:**
```python
# Your existing implementation
class MixtureOfExperts(nn.Module):
    # Standard MLP experts
    # Simple top-k routing
    # Basic load balancing
```

**DeepseekV3MoE:**
```python
# DeepSeek's implementation
class DeepseekV3MoE(nn.Module):
    # DeepseekV3MLP experts (gated)
    # Advanced MoEGate with sigmoid scoring
    # Group-based expert selection
```

## Fair Comparison Setup

Both models use:
- **Same architecture**: 256d, 3L, 4H
- **Same MoE configuration**: 8 experts, top-2 selection
- **Same training setup**: 1000 steps, batch size 16
- **Same dataset**: 100,000 tokens
- **Same attention mechanism**: Standard multi-head attention
- **Same normalization**: Standard RMSNorm

## Files

- `exp5_baseline_moe_model.py` - Baseline model with your MoE implementation
- `exp5_deepseek_moe_model.py` - Model using DeepseekV3MoE
- `exp5_trainer.py` - Training script that compares both models
- `README.md` - This documentation

## Running the Experiment

```bash
cd /root/blueberry-llm-kimi-deepseek/experiments/exp5
python exp5_trainer.py
```

## Expected Results

The experiment will:
1. Train both models for 1000 steps
2. Evaluate performance on validation set
3. Compare validation loss, accuracy, and perplexity
4. Generate loss vs time plots
5. Save results to `exp5_results/exp5_moe_comparison.json`

## Configuration

- **Training Steps**: 1000
- **Batch Size**: 16
- **Model Size**: 256d, 3L, 4H
- **MoE**: 8 experts, top-2 selection
- **Sequence Length**: 256
- **Evaluation**: Every 100 steps

## Import Strategy

This experiment follows the "import as much as possible" approach:
- Imports `DeepseekV3MoE` directly from `deepseek_modeling.py`
- Uses existing MoE infrastructure from `models.components`
- Reuses training utilities from `training.trainer`
- Leverages data loading from `data.loader`

## Expected Training Time

- **Per Model**: ~5-10 minutes
- **Total Experiment**: ~10-20 minutes

The experiment is designed to be fair and unbiased, comparing only the MoE implementations while keeping everything else identical.
