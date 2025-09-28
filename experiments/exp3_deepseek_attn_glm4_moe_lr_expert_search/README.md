# Experiment 3: DeepSeek Attention + GLM4 MoE with Learning Rate & Expert Search

## Overview
This experiment focuses on optimizing and training the DeepSeek Attention + GLM4 MoE model through comprehensive learning rate search and expert configuration search. The goal is to find the optimal hyperparameters for the DeepSeek Attention + GLM4 MoE architecture and then train it for extended periods.

## Model Architecture
**DeepSeek Attention + GLM4 MoE 512d** model:
- **Hidden Size**: 512 dimensions
- **Attention Heads**: 8 heads (DeepSeek Attention)
- **Hidden Layers**: 12 layers
- **Intermediate Size**: 1024 (smaller for MoE efficiency)
- **MoE Configuration**: Optimized via expert search
- **Parameters**: ~145M (varies with expert configuration)

## Key Features

### 🔍 Learning Rate Search
- **Multiple LRs**: Test 1e-4, 3e-4, 1e-3, 3e-3
- **Quick Evaluation**: 1000 steps per learning rate
- **Comprehensive Metrics**: Loss, accuracy, perplexity tracking
- **Visualization**: Comparison plots for all learning rates
- **Recommendation**: Automatic best LR selection

### 🧠 Expert Configuration Search
- **Expert Counts**: Test 4, 8, 16 experts
- **Top-k Values**: Test top-1, top-2 routing
- **Combinations**: 6 different expert configurations
- **Performance Analysis**: Comprehensive comparison plots
- **Recommendation**: Optimal expert setup selection

### 🚀 Extended Training
- **Long-term Training**: 10,000+ steps with optimal configurations
- **Regular Checkpoints**: Save model every 3,000 steps
- **Frequent Evaluation**: Evaluate every 100 steps
- **HellaSwag Benchmark**: Run benchmark every 1,000 steps
- **Progress Tracking**: Comprehensive loss curve visualization

## Usage

### Step 1: Learning Rate Search
```bash
cd experiments/exp3_deepseek_attn_glm4_moe_lr_expert_search
python lr_search.py
```

### Step 2: Expert Configuration Search
```bash
python expert_search.py
```

### Step 3: Extended Training (after optimizations)
```bash
python trainer.py
```

## Search Process

### Learning Rate Search
The learning rate search tests multiple learning rates:
- **1e-4**: Conservative learning rate
- **3e-4**: Medium-low learning rate  
- **1e-3**: Medium-high learning rate
- **3e-3**: Aggressive learning rate

### Expert Configuration Search
The expert search tests different MoE configurations:
- **4 experts, top-1**: Minimal MoE setup
- **4 experts, top-2**: Balanced 4-expert setup
- **8 experts, top-1**: Standard MoE setup
- **8 experts, top-2**: Balanced 8-expert setup
- **16 experts, top-1**: High-capacity MoE setup
- **16 experts, top-2**: Maximum MoE setup

### Analysis & Recommendations
Both searches provide:
- **Performance Comparison**: Which configuration gives best results
- **Training Stability**: Which configuration trains most stably
- **Convergence Speed**: Which configuration converges fastest
- **Resource Efficiency**: Balance between performance and compute

## Training Configuration

### Learning Rate Search Settings
- **Steps per LR**: 1000 steps
- **Evaluation**: Every 100 steps
- **Batch Size**: 128
- **Model**: DeepSeek Attention + GLM4 MoE 512d

### Expert Search Settings
- **Steps per Config**: 800 steps
- **Evaluation**: Every 100 steps
- **Batch Size**: 128
- **Learning Rate**: 1e-3 (fixed for fair comparison)

### Extended Training Settings
- **Total Steps**: 10,000+
- **Checkpoint Every**: 3,000 steps
- **Evaluation Every**: 100 steps
- **HellaSwag Benchmark**: Every 1,000 steps
- **Batch Size**: 16
- **Learning Rate**: Optimal LR from search
- **Expert Config**: Optimal config from search

## Expected Results

### Learning Rate Search Results
- **Best LR**: Typically 1e-3 or 3e-4
- **Validation Loss Range**: 0.01 - 0.05
- **Training Time**: ~30 minutes for full search
- **Comparison Plot**: Visual comparison of all LRs

### Expert Search Results
- **Best Config**: Typically 8 experts, top-2
- **Performance Range**: Varies significantly by config
- **Training Time**: ~45 minutes for full search
- **Comparison Plots**: Multiple visualization plots

### Extended Training Results
- **Final Validation Loss**: < 0.01
- **Final Validation Accuracy**: > 99.8%
- **Final Perplexity**: < 1.01
- **Training Time**: ~2-3 hours for 10k steps

## File Structure
```
exp3_deepseek_attn_glm4_moe_lr_expert_search/
├── lr_search.py              # Learning rate search script
├── expert_search.py          # Expert configuration search script
├── trainer.py               # Extended training script
├── README.md               # This documentation
├── lr_search_results/      # Generated during LR search
│   ├── lr_search_results.json
│   ├── lr_search_comparison.png
│   ├── lr_recommendation.json
│   └── lr_1.00e-03_result.json
├── expert_search_results/   # Generated during expert search
│   ├── expert_search_results.json
│   ├── expert_search_comparison.png
│   ├── expert_recommendation.json
│   └── experts_8_top_2_result.json
└── exp3_results/           # Generated during extended training
    ├── exp3_extended_results.json
    ├── exp3_training_curves.png
    ├── final_model.pt
    ├── checkpoint_step_3000.pt
    └── hellaswag_benchmark/
        └── step_1000_hellaswag_results.json
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

### 2. Expert Configuration Search
```bash
python expert_search.py
```
This will:
- Test 6 different expert configurations
- Generate multiple comparison plots
- Save results and recommendation
- Take ~45 minutes

### 3. Review Results
Check the recommendation files:
- `lr_search_results/lr_recommendation.json`
- `expert_search_results/expert_recommendation.json`

### 4. Extended Training
```bash
python trainer.py
```
This will:
- Use the recommended learning rate and expert config
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

### Expert Search Output
```
🧪 Training with 8 experts, top-2
  Step 100: Train Loss=7.1234, Val Loss=5.2345, Val Acc=0.3456
  Step 200: Train Loss=6.1234, Val Loss=4.2345, Val Acc=0.4567
  ...

🏆 Best Expert Configuration Found:
   Experts: 8
   Top-k: 2
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
💾 Checkpoint saved: exp3_results/checkpoint_step_3000.pt
```

## Hardware Requirements
- **GPU**: NVIDIA GPU with 8+ GB VRAM recommended
- **RAM**: 16+ GB system RAM
- **Storage**: 3+ GB for checkpoints and results

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size in config
2. **LR Search Too Slow**: Reduce max_steps in lr_search.py
3. **Expert Search Too Slow**: Reduce max_steps in expert_search.py
4. **Poor Results**: Try different LR ranges or expert configs

### Performance Tips
1. **Use GPU**: Ensure CUDA is available
2. **Monitor Memory**: Watch GPU memory usage
3. **Run Searches First**: Always run LR and expert searches first
4. **Use Recommendations**: Don't skip the optimization steps

## Comparison with Other Experiments

| Aspect | Experiment 1 | Experiment 2 | Experiment 3 |
|--------|---------------|--------------|---------------|
| **Purpose** | Architecture comparison | LR optimization | LR + Expert optimization |
| **Model** | Multiple architectures | DeepSeek Attn+MLP | DeepSeek Attn+GLM4 MoE |
| **Optimization** | None | LR search only | LR + Expert search |
| **Training Steps** | 1,500 | 10,000+ | 10,000+ |
| **Complexity** | Low | Medium | High |

## Next Steps
After completing Experiment 3:
1. **Analyze Results**: Compare LR and expert search results
2. **Compare Performance**: Compare with Experiments 1 and 2
3. **Extend Training**: Run even longer if needed
4. **Fine-tune**: Adjust other hyperparameters
5. **Deploy**: Use optimized model for applications

This experiment provides a complete pipeline for optimizing and training the DeepSeek Attention + GLM4 MoE architecture with the best possible learning rate and expert configuration.
