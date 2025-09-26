# Experiment 1: DeepSeek Attention Integration

## Overview

This experiment integrates DeepSeek's advanced attention mechanisms into our MoE model using the **original DeepSeek implementation** from `deepseek_modeling.py`. This ensures correctness and minimizes custom code that could introduce bugs.

## Research Design & Fairness

### What We're Testing
We're comparing three configurations to evaluate the impact of DeepSeek's attention innovations:

1. **Baseline**: Standard multi-head attention (our original MoE model)
2. **LoRA**: DeepSeek attention with LoRA-style Q/K/V projections
3. **Enhanced**: DeepSeek attention with LoRA + separate head dimensions + RoPE scaling + attention bias

### Why This is Fair Research
- **Same Architecture**: All models use identical MoE structure, only attention differs
- **Same Training**: Identical training procedure, data, and hyperparameters
- **Same Evaluation**: Same metrics and evaluation protocol
- **Original Implementation**: Uses DeepSeek's actual code, not custom reimplementation
- **Controlled Variables**: Only attention mechanism varies between experiments

### DeepSeek Attention Features Being Tested

#### 1. LoRA-style Q/K/V Projections
- **Q Projection**: Optional LoRA-style projection with configurable rank
- **KV Projection**: LoRA-style projection with MQA (Multi-Query Attention) support
- **Hypothesis**: Reduced parameters without performance loss

#### 2. Separate Head Dimensions
- **QK Head Dim**: Configurable dimension for query/key projections
- **V Head Dim**: Configurable dimension for value projections
- **Hypothesis**: Better optimization of different attention components

#### 3. Advanced RoPE Scaling
- **Linear Scaling**: Simple linear scaling of RoPE frequencies
- **Hypothesis**: Better handling of longer sequences

#### 4. Enhanced Attention Bias
- **Configurable Bias**: Optional bias in attention projections
- **Hypothesis**: Improved model flexibility

## Experiment Configurations

### Baseline Configuration
- **Model**: Original MoE model with standard attention
- **Purpose**: Control group to establish baseline performance
- **Features**: None (standard multi-head attention)

### LoRA Configuration
- **Model**: DeepSeek MoE model with LoRA projections
- **Purpose**: Test efficiency gains from LoRA-style projections
- **Features**: Q LoRA rank=32, KV LoRA rank=64

### Enhanced Configuration
- **Model**: DeepSeek MoE model with all available features
- **Purpose**: Test combined benefits of all DeepSeek innovations
- **Features**: LoRA + separate head dims + RoPE scaling + attention bias

## Research Questions

1. **Efficiency**: Do LoRA projections reduce parameters without hurting performance?
2. **Optimization**: Do separate head dimensions improve attention quality?
3. **Scaling**: Does RoPE scaling help with sequence length handling?
4. **Combined Effect**: What's the cumulative benefit of all features?

## Implementation Details

### Using Original DeepSeek Code
- **Attention**: `DeepseekV3Attention` from `deepseek_modeling.py`
- **Normalization**: `DeepseekV3RMSNorm` from `deepseek_modeling.py`
- **Configuration**: `DeepseekV3Config` from `configuration_deepseek.py`
- **MoE**: Our existing `MixtureOfExperts` implementation

### Architecture
- **Transformer Blocks**: DeepSeek attention + our MoE feed-forward
- **Embeddings**: Standard token embeddings
- **Output**: Standard language modeling head
- **Training**: Identical procedure for all configurations

## Running the Experiment

```bash
# Run full experiment
python experiments/exp1_trainer_import.py

# Run specific configurations
python -c "
from experiments.exp1_trainer_import import Experiment1ImportTrainer
from configs.moe_config import MoEModelConfig
trainer = Experiment1ImportTrainer(MoEModelConfig())
results = trainer.run_experiment(['baseline', 'lora', 'enhanced'])
"
```

## Expected Outcomes

### Performance Metrics
- **Validation Loss**: Lower is better
- **Validation Accuracy**: Higher is better
- **Validation Perplexity**: Lower is better
- **Training Time**: Efficiency measure (minutes)
- **Peak Memory Usage**: Memory efficiency (GB)
- **Memory Used**: Actual memory consumption (GB)

### Success Criteria
- **LoRA**: Should maintain or improve performance with fewer parameters
- **Enhanced**: Should show best overall performance
- **Baseline**: Establishes performance floor

### Efficiency Analysis
- **Memory Efficiency**: Compare peak memory usage across configurations
- **Training Speed**: Compare training time per epoch
- **Parameter Efficiency**: Compare model size vs performance
- **Resource Utilization**: GPU/CPU usage patterns

## Files Structure

```
experiments/
├── __init__.py
├── exp1_README.md
├── exp1_deepseek_import.py      # DeepSeek MoE model using original components
├── exp1_config_import.py        # Experiment configurations
├── exp1_trainer_import.py       # Training and comparison script
└── exp1_import_results/         # Results directory (created during run)
    └── experiment1_import_results.json
```

## Dependencies

- PyTorch
- Transformers (for DeepSeek components)
- Flash Attention (optional, for future experiments)
- Existing MoE model components
- Training utilities

## Notes

- **Reproducible**: Fixed seeds and identical training procedures
- **Fair Comparison**: Only attention mechanism varies
- **Original Implementation**: Uses DeepSeek's actual code
- **Minimal Custom Code**: Maximum reuse of proven implementations
