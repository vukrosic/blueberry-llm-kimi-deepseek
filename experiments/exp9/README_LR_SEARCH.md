# Learning Rate Search for Experiment 9

This directory contains learning rate search implementations for Experiment 9, which focuses on finding the optimal learning rate for the Attention+MLP 512d model.

## Files

- `lr_search_exp9.py` - Main learning rate search implementation
- `run_lr_search_exp9.py` - Complete runner that does LR search + long-term training
- `quick_lr_search_exp9.py` - Quick LR search for testing (fewer steps)
- `exp9_trainer.py` - Original long-term trainer
- `run_hellaswag_benchmark.py` - HellaSwag benchmark runner

## Usage

### Quick Learning Rate Search (Recommended for Testing)

Run a quick learning rate search with fewer training steps:

```bash
python quick_lr_search_exp9.py
```

This will:
- Test 4 learning rates: [1e-4, 3e-4, 1e-3, 3e-3]
- Train for 1000 steps per learning rate
- Evaluate every 50 steps
- Use early stopping with patience of 3
- Save results to `exp9_quick_lr_search/`

### Full Learning Rate Search

Run a comprehensive learning rate search:

```bash
python lr_search_exp9.py
```

This will:
- Test 7 learning rates: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
- Train for 2000 steps per learning rate
- Evaluate every 100 steps
- Use early stopping with patience of 5
- Save results to `exp9_lr_search/`

### Complete Pipeline (LR Search + Long-term Training)

Run the complete pipeline that finds the best LR and then trains long-term:

```bash
python run_lr_search_exp9.py
```

This will:
1. Run comprehensive learning rate search
2. Find the best learning rate
3. Run long-term training (10k steps) with the best LR
4. Save results to both `exp9_lr_search/` and `exp9_results_best_lr/`

## Learning Rate Search Features

### Systematic Search
- Tests multiple learning rates across a wide range
- Uses the same model architecture and training setup for fair comparison
- Implements proper learning rate scheduling with warmup and cosine decay

### Early Stopping
- Monitors validation loss during training
- Stops training early if validation loss doesn't improve
- Prevents overfitting and saves computational resources

### Comprehensive Evaluation
- Tracks training loss, validation loss, and validation accuracy
- Records learning rate schedules
- Measures training time and efficiency

### Visualization
- Creates comparison plots showing:
  - Validation loss curves for all learning rates
  - Final validation loss vs learning rate
  - Training loss curves
  - Learning rate schedules

### Results Analysis
- Automatically finds the best learning rate based on validation loss
- Provides detailed metrics for each learning rate tested
- Saves recommendations for long-term training

## Output Structure

```
exp9_lr_search/                    # Full LR search results
├── results/                       # Individual LR results
│   ├── lr_1.00e-04_result.json
│   ├── lr_3.00e-04_result.json
│   └── ...
├── lr_search_results.json        # All results summary
└── lr_search_comparison.png      # Comparison plots

exp9_quick_lr_search/              # Quick LR search results
├── results/                       # Individual LR results
├── lr_search_results.json        # All results summary
├── lr_recommendation.json        # Best LR recommendation
└── lr_search_comparison.png      # Comparison plots

exp9_results_best_lr/             # Long-term training with best LR
├── checkpoints/                  # Model checkpoints
├── hellaswag_benchmark/          # HellaSwag evaluation results
├── exp9_long_term_results.json   # Training results
└── exp9_long_term_training_curves.png  # Training curves
```

## Configuration

The learning rate search uses the following model configuration:

```python
MoEModelConfig(
    d_model=512,           # Model dimension
    n_heads=8,             # Number of attention heads
    n_layers=12,           # Number of transformer layers
    d_ff=2048,             # Feed-forward dimension
    num_experts=8,         # Number of experts
    expert_top_k=2,        # Top-k experts
    batch_size=128,        # Batch size
    max_seq_len=256,       # Sequence length
    max_tokens=100000,     # Maximum tokens
    num_documents=1000,    # Number of documents
)
```

## Learning Rate Schedule

Each training run uses a learning rate schedule with:
- **Warmup**: Linear warmup for 5% of total steps
- **Decay**: Cosine decay from peak LR to 10% of peak LR
- **Formula**: `lr = peak_lr * (0.1 + 0.9 * 0.5 * (1 + cos(π * progress)))`

## Early Stopping

Early stopping is implemented to prevent overfitting:
- Monitors validation loss every evaluation step
- Stops training if validation loss doesn't improve for N consecutive evaluations
- Default patience: 3 (quick search) or 5 (full search)

## Integration with Existing Code

The learning rate search integrates seamlessly with the existing Experiment 9 infrastructure:
- Uses the same model architecture (`AttentionMLP_512dModel`)
- Compatible with existing data loading and preprocessing
- Can be used to find optimal LR for the long-term trainer
- Results can be loaded into the HellaSwag benchmark evaluator

## Recommendations

1. **Start with Quick Search**: Run `quick_lr_search_exp9.py` first to get a rough idea of the optimal learning rate range
2. **Use Full Search**: Run `lr_search_exp9.py` for comprehensive analysis
3. **Long-term Training**: Use the complete pipeline `run_lr_search_exp9.py` for production training
4. **Monitor Results**: Check the generated plots and JSON files for detailed analysis
5. **Customize**: Modify the learning rate ranges in the scripts based on your specific needs
