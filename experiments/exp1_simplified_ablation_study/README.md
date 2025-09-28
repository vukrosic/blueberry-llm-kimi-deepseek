# Experiment 1: Simplified Ablation Study

## Overview
This experiment focuses on a simplified set of ablations centered around the 512 hidden dimension scale, matching the architecture from the provided JSON config. The goal is to test meaningful combinations while keeping the scope manageable with just 5 models.

## Architecture Target
Based on the provided JSON config:
- **Hidden Size**: 576 (we use 512 as base scale)
- **Attention Heads**: 9 (we use 8 for efficiency)
- **Hidden Layers**: 30 (we use 3 for efficiency)
- **Intermediate Size**: 1536 (we use 2048 for MLP models)
- **Experts**: 8 experts (matches target architecture)
- **Top-k**: 2 (consistent across MoE models)

## Models (5 total)

### Baseline (1 model)
- `baseline`: Standard MoE model without DeepSeek components

### MLP (1 model)
- `mlp_512d`: DeepSeek MLP with 512 dimensions (target scale, 2048d inner)

### Attention+MLP (1 model)
- `attention_mlp_512d`: DeepSeek Attention + MLP with 512 dimensions (target scale, 2048d inner)

### MoE (1 model)
- `moe_8e_2k_512d`: GLM4 MoE with 8 experts, top-2, 512d (target scale)

### Attention+MoE (1 model)
- `attention_moe_8e_2k_512d`: DeepSeek Attention + GLM4 MoE with 8 experts, top-2, 512d (target scale)

## Key Features
- **Reduced Scope**: Only 5 models vs 32 in Experiment 6
- **512 Scale Focus**: All models centered around 512 hidden dimensions
- **Standard MLP Scaling**: 512d → 2048d inner dimension (4x scaling)
- **Single MoE Configuration**: 8 experts, top-2 (matches target architecture)
- **Target Architecture**: One model matches the provided JSON config exactly
- **Minimal Code**: Streamlined implementation for quick testing
- **HellaSwag Benchmark**: Automatic evaluation on HellaSwag benchmark at the end

## Usage
```bash
cd experiments/exp1_simplified_ablation_study
python exp1_trainer.py
```

## Expected Results
- Quick comparison of different scaling approaches
- Clear performance differences between MLP and MoE at 512 scale
- Understanding of how DeepSeek components perform at target architecture size
- Reduced computational cost compared to full ablation studies
- HellaSwag benchmark scores for all successful models

## HellaSwag Benchmark Integration
The experiment automatically evaluates all successfully trained models on the HellaSwag benchmark at the end of training. This provides:

- **Standardized Evaluation**: Consistent benchmark across all model variants
- **Minimal Code**: Uses `lm-evaluation-harness` for reliable evaluation
- **Automatic Integration**: No manual intervention required
- **Results Storage**: Benchmark results saved alongside training metrics

### Benchmark Results Location
- Individual model results: `exp1_results/hellaswag_benchmark/{model_name}_hellaswag_results.json`
- Combined results: `exp1_results/hellaswag_benchmark/all_models_hellaswag_results.json`
- Integrated results: `exp1_results/exp1_reduced_results.json` (includes benchmark scores)
