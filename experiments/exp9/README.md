# Experiment 9: DeepSeek Attention + MLP with Learning Rate Search

## Overview
This experiment focuses on training the DeepSeek Attention + MLP model with learning rate optimization. The goal is to find the optimal learning rate for the DeepSeek Attention + MLP architecture and then train it for extended periods.

## Model Architecture
**DeepSeek Attention + MLP 512d** model:
- **Hidden Size**: 512 dimensions
- **Attention Heads**: 8 heads
- **Hidden Layers**: 12 layers
- **Intermediate Size**: 2048 (4x d_model scaling)
- **Parameters**: ~145M

## Key Features

### 🔍 Learning Rate Search
- **Multiple LRs**: Test 1e-4, 3e-4, 1e-3, 3e-3
- **Quick Evaluation**: 1000 steps per learning rate
- **Comprehensive Metrics**: Loss, accuracy, perplexity tracking
- **Visualization**: Comparison plots for all learning rates
- **Recommendation**: Automatic best LR selection

### 🚀 Extended Training
- **Long-term Training**: 10,000+ steps with optimal LR
- **Regular Checkpoints**: Save model every 3,000 steps
- **Frequent Evaluation**: Evaluate every 100 steps
- **HellaSwag Benchmark**: Run benchmark every 1,000 steps
- **Progress Tracking**: Comprehensive loss curve visualization

## Usage

### Learning Rate Search
```bash
cd experiments/exp9
python lr_search.py
```

### Extended Training (after LR search)
```bash
cd experiments/exp9
python exp9_trainer.py
```

## Learning Rate Search Process

### Step 1: Run LR Search
The learning rate search tests multiple learning rates:
- **1e-4**: Conservative learning rate
- **3e-4**: Medium-low learning rate  
- **1e-3**: Medium-high learning rate
- **3e-3**: Aggressive learning rate

### Step 2: Analysis
The search provides:
- **Validation Loss Comparison**: Which LR gives lowest loss
- **Validation Accuracy Comparison**: Which LR gives highest accuracy
- **Training Stability**: Which LR trains most stably
- **Convergence Speed**: Which LR converges fastest

### Step 3: Recommendation
The system automatically recommends the best learning rate based on validation loss.

## Training Configuration

### Learning Rate Search Settings
- **Steps per LR**: 1000 steps
- **Evaluation**: Every 100 steps
- **Batch Size**: 128
- **Model**: DeepSeek Attention + MLP 512d

### Extended Training Settings
- **Total Steps**: 10,000+
- **Checkpoint Every**: 3,000 steps
- **Evaluation Every**: 100 steps
- **HellaSwag Benchmark**: Every 1,000 steps
- **Batch Size**: 16
- **Learning Rate**: Optimal LR from search

## Expected Results

### Learning Rate Search Results
- **Best LR**: Typically 1e-3 or 3e-4
- **Validation Loss Range**: 0.01 - 0.05
- **Training Time**: ~30 minutes for full search
- **Comparison Plot**: Visual comparison of all LRs

### Extended Training Results
- **Final Validation Loss**: < 0.01
- **Final Validation Accuracy**: > 99.8%
- **Final Perplexity**: < 1.01
- **Training Time**: ~2-3 hours for 10k steps

## File Structure
```
experiments/exp9/
├── lr_search.py              # Learning rate search script
├── exp9_trainer.py          # Extended training script
├── README.md               # This documentation
├── lr_search_results/      # Generated during LR search
│   ├── lr_search_results.json
│   ├── lr_search_comparison.png
│   ├── lr_recommendation.json
│   └── lr_1.00e-03_result.json
└── exp9_results/           # Generated during training
    ├── exp9_long_term_results.json
    ├── exp9_long_term_training_curves.png
    └── hellaswag_benchmark/
        └── attention_mlp_512d_hellaswag_results.json
```

## Workflow

### 1. Learning Rate Search
```bash
python lr_search.py
```
This will:
- Test 4 different learning rates
- Generate comparison plots
- Save results and recommendation
- Take ~30 minutes

### 2. Review Results
Check `lr_search_results/lr_recommendation.json` for the recommended learning rate.

### 3. Extended Training
```bash
python exp9_trainer.py
```
This will:
- Use the recommended learning rate
- Train for 10,000+ steps
- Save regular checkpoints
- Run HellaSwag benchmarks
- Take ~2-3 hours

## Monitoring Progress

### Learning Rate Search Output
```
🧪 Training with LR=1.00e-04
  Step 100: Train Loss=8.2341, Val Loss=6.1234, Val Acc=0.2345
  Step 200: Train Loss=7.1234, Val Loss=5.2345, Val Acc=0.3456
  ...

🏆 Best Learning Rate Found:
   Learning Rate: 1.00e-03
   Validation Loss: 0.012345
   Validation Accuracy: 0.998765
```

### Extended Training Output
```
Step 0/10000: Loss=10.9205
Step 100/10000: Loss=8.1234
   Val Loss: 0.1234, Val Acc: 0.9876
Step 200/10000: Loss=6.2345
   Val Loss: 0.0987, Val Acc: 0.9923
💾 Checkpoint saved: exp9_results/checkpoints/checkpoint_step_3000.pt
```

## Hardware Requirements
- **GPU**: NVIDIA GPU with 8+ GB VRAM recommended
- **RAM**: 16+ GB system RAM
- **Storage**: 2+ GB for checkpoints and results

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size in config
2. **LR Search Too Slow**: Reduce max_steps in lr_search.py
3. **Poor LR Results**: Try different LR ranges

### Performance Tips
1. **Use GPU**: Ensure CUDA is available
2. **Monitor Memory**: Watch GPU memory usage
3. **Start with LR Search**: Always run LR search first
4. **Use Recommended LR**: Don't skip the LR recommendation

## Comparison with Experiment 8

| Aspect | Experiment 8 | Experiment 9 |
|--------|---------------|--------------|
| **Purpose** | Architecture comparison | LR optimization + extended training |
| **Training Steps** | 1,500 | 10,000+ |
| **Learning Rate** | Fixed | Optimized via search |
| **Checkpoints** | None | Every 3,000 steps |
| **Monitoring** | Basic | Comprehensive |

## Next Steps
After completing Experiment 9:
1. **Analyze LR Results**: Understand which LR works best
2. **Compare Performance**: Compare with Experiment 8 results
3. **Extend Training**: Run even longer if needed
4. **Fine-tune**: Adjust other hyperparameters
5. **Deploy**: Use optimized model for applications

This experiment provides a complete pipeline for optimizing and training the DeepSeek Attention + MLP architecture with the best possible learning rate.