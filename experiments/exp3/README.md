# Experiment 3: Mamba State Space Model

## Overview

This experiment tests the **Mamba architecture** as an alternative to attention mechanisms in transformer models. Mamba uses Structured State Space Models (SSMs) to process sequences efficiently without relying on attention mechanisms.

## Key Features

- **No Attention Mechanisms**: Uses state space models instead of self-attention
- **Efficient Long Sequences**: Better computational complexity for long sequences
- **MoE Integration**: Combines Mamba blocks with Mixture of Experts
- **Multiple Configurations**: Tests different model sizes and architectures

## Architecture

### Mamba SSM Block
- **State Space Modeling**: Uses structured state space models for sequence processing
- **Convolutional Processing**: 1D convolution for local dependencies
- **Gating Mechanism**: Data-controlled gating for selective information flow
- **Efficient Computation**: Linear complexity with respect to sequence length

### Model Variants

1. **Mamba-Small**: 256d, 4L, 4H, 1024ff
2. **Mamba-Medium**: 512d, 6L, 8H, 2048ff  
3. **Mamba-Large**: 768d, 8L, 12H, 3072ff
4. **Mamba-Wide**: 512d, 4L, 16H, 4096ff
5. **Mamba-Deep**: 384d, 12L, 8H, 1536ff

## Files

- `exp3_mamba_trainer.py`: Main experiment script
- `exp3_config_import.py`: Configuration management
- `README.md`: This documentation

## Running the Experiment

```bash
cd experiments/exp3
python exp3_mamba_trainer.py
```

## Expected Results

The experiment will test:
- Training efficiency compared to attention-based models
- Memory usage patterns
- Convergence speed
- Final model performance
- Scalability with different model sizes

## Key Differences from Attention

1. **Computational Complexity**: O(L) vs O(L²) for sequence length L
2. **Memory Usage**: More efficient for long sequences
3. **Parallelization**: Different parallelization patterns
4. **Information Flow**: State-based vs attention-based information flow

## Comparison with Previous Experiments

- **Exp1**: DeepSeek attention mechanisms
- **Exp2**: Architecture search with attention variants
- **Exp3**: Mamba SSM without attention (this experiment)

This experiment provides a baseline for comparing attention-based and non-attention-based approaches in the same experimental setup.
