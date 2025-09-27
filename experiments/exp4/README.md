# Experiment 4: DeepseekV3MLP vs Baseline MLP Comparison

## Overview
This experiment compares the performance of DeepseekV3MLP (from deepseek_modeling.py) against the baseline MLP implementation in a MoE (Mixture of Experts) transformer architecture.

## Models Compared

### Baseline MLP Model
- Uses standard PyTorch MLP layers
- Standard RMSNorm normalization
- Standard transformer block architecture

### DeepseekV3MLP Model  
- Uses DeepseekV3MLP from deepseek_modeling.py
- Implements the specialized MLP architecture from DeepSeek V3
- Uses SiLU activation function
- Maintains same MoE structure for fair comparison

## Key Differences

The main difference is in the MLP implementation:

**Baseline MLP:**
```python
# Standard MLP in transformer block
mlp = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Linear(d_ff, d_model)
)
```

**DeepseekV3MLP:**
```python
# DeepseekV3MLP with specialized architecture
self.deepseek_mlp = DeepseekV3MLP(deepseek_config)
# Uses gate_proj, up_proj, down_proj with SiLU activation
```

## Files

- `exp4_baseline_model.py` - Baseline model with standard MLP
- `exp4_deepseek_mlp_model.py` - Model using DeepseekV3MLP
- `exp4_trainer.py` - Training script that compares both models
- `README.md` - This documentation

## Running the Experiment

```bash
cd /root/blueberry-llm-kimi-deepseek/experiments/exp4
python exp4_trainer.py
```

## Expected Results

The experiment will:
1. Train both models for 1000 steps
2. Evaluate performance on validation set
3. Compare validation loss, accuracy, and perplexity
4. Generate loss vs time plots
5. Save results to `exp4_results/exp4_mlp_comparison.json`

## Configuration

- **Training Steps**: 1000 (5x longer than minimal)
- **Batch Size**: 16
- **Model Size**: 256d, 3L, 4H
- **Sequence Length**: 256
- **Evaluation**: Every 100 steps

## Import Strategy

This experiment follows the "import as much as possible" approach:
- Imports `DeepseekV3MLP` directly from `deepseek_modeling.py`
- Uses existing MoE infrastructure from `models.layers`
- Reuses training utilities from `training.trainer`
- Leverages data loading from `data.loader`

## Expected Training Time

- **Per Model**: ~5-10 minutes
- **Total Experiment**: ~10-20 minutes

The experiment is designed to be fast but comprehensive enough to show meaningful differences between the MLP implementations.
