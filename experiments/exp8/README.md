# Experiment 8: Reduced Ablation Study

## Overview
This experiment focuses on a reduced set of ablations centered around the 512 hidden dimension scale, matching the architecture from the provided JSON config. The goal is to test meaningful combinations with powers of 2 ablations while keeping the scope manageable.

## Architecture Target
Based on the provided JSON config:
- **Hidden Size**: 576 (we use 512 as base scale)
- **Attention Heads**: 9 (we use powers of 2: 4, 8, 16)
- **Hidden Layers**: 30 (we use 3 for efficiency)
- **Intermediate Size**: 1536 (we use powers of 2: 1024, 2048, 4096)
- **Experts**: Powers of 2 (4, 8, 16)
- **Top-k**: 2 (consistent across MoE models)

## Models (9 total)

### Baseline (1 model)
- `baseline`: Standard MoE model without DeepSeek components

### MLP (1 model)
- `mlp_512d`: DeepSeek MLP with 512 dimensions (target scale, 2048d inner)

### Attention+MLP (1 model)
- `attention_mlp_512d`: DeepSeek Attention + MLP with 512 dimensions (target scale, 2048d inner)

### MoE Expert Scaling (3 models)
- `moe_4e_2k_512d`: GLM4 MoE with 4 experts, top-2, 512d
- `moe_8e_2k_512d`: GLM4 MoE with 8 experts, top-2, 512d (target scale)
- `moe_16e_2k_512d`: GLM4 MoE with 16 experts, top-2, 512d

### Attention+MoE Scaling (3 models)
- `attention_moe_4e_2k_512d`: DeepSeek Attention + GLM4 MoE with 4 experts, top-2, 512d
- `attention_moe_8e_2k_512d`: DeepSeek Attention + GLM4 MoE with 8 experts, top-2, 512d (target scale)
- `attention_moe_16e_2k_512d`: DeepSeek Attention + GLM4 MoE with 16 experts, top-2, 512d

## Key Features
- **Reduced Scope**: Only 9 models vs 32 in Experiment 6
- **512 Scale Focus**: All models centered around 512 hidden dimensions
- **Standard MLP Scaling**: 512d → 2048d inner dimension (4x scaling)
- **Powers of 2 Experts**: Consistent use of powers of 2 for experts (4, 8, 16)
- **Target Architecture**: One model matches the provided JSON config exactly
- **Minimal Code**: Streamlined implementation for quick testing

## Usage
```bash
cd experiments/exp8
python exp8_trainer.py
```

## Expected Results
- Quick comparison of different scaling approaches
- Clear performance differences between MLP and MoE at 512 scale
- Understanding of how DeepSeek components perform at target architecture size
- Reduced computational cost compared to full ablation studies
