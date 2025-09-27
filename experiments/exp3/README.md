# Experiment 3: RMSNorm Comparison

## Overview
Minimal experiment comparing baseline RMSNorm vs DeepSeek RMSNorm with minimal code changes.

## Files
- `exp3_baseline_model.py` - Baseline model using standard PyTorch RMSNorm
- `exp3_deepseek_model.py` - DeepSeek model using DeepseekV3RMSNorm
- `exp3_trainer.py` - Minimal trainer to compare both models

## Key Changes
- **Baseline**: Uses `nn.RMSNorm(config.d_model)`
- **DeepSeek**: Uses `DeepseekV3RMSNorm(config.d_model, eps=1e-6)`
- Everything else is identical

## Usage
```bash
cd /root/blueberry-llm-kimi-deepseek
python experiments/exp3/exp3_trainer.py
```

## Results (1000 steps, 5x longer training)

### Performance Comparison
| Model | Val Loss | Val Acc | Val Perp | Time (min) | Params (M) |
|-------|----------|---------|----------|------------|------------|
| Baseline RMSNorm | 2.0551 | 58.74% | 7.81 | 1.21 | 25.96 |
| DeepSeek RMSNorm | 2.0058 | 59.20% | 7.43 | 1.18 | 25.96 |

### Key Findings
- **DeepSeek RMSNorm is 2.40% better** than baseline RMSNorm
- **Faster training**: 1.18 min vs 1.21 min (2.5% faster)
- **Better accuracy**: 59.20% vs 58.74% (+0.46%)
- **Lower perplexity**: 7.43 vs 7.81 (5% better)
- **Identical parameters**: Both models have 25.96M parameters

### Training Progress
- Both models show steady improvement over 1000 steps
- DeepSeek maintains consistent advantage throughout training
- Loss curves show DeepSeek consistently outperforming baseline
- The advantage becomes more pronounced with longer training

## Generated Files
- `exp3_rmsnorm_comparison.json` - Detailed results data
- `loss_vs_time_comparison.png` - Loss vs time plot showing both models

## Configuration
- Training: 1000 steps (5x longer), small model (256d, 3L, 4H)
- Small dataset: 100K tokens, 1K documents
- Runtime: ~2.5 minutes total
- Evaluation: Every 100 steps

## Conclusion
DeepSeek RMSNorm shows consistent improvement over baseline RMSNorm with:
- Better performance (2.40% improvement)
- Faster training (2.5% faster)
- Same parameter count
- Minimal code changes (only RMSNorm implementation differs)
