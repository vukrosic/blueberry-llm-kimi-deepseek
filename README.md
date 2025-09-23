# Blueberry LLM 🫐 - T4 Optimized

A Tesla T4 GPU-optimized Mixture of Experts (MoE) language model implementation.

**Goal: Make LLM training accessible on T4 GPUs** - optimized specifically for Tesla T4 GPU performance with automatic configuration and hardware optimization to create state of the art LLM on single T4 GPU.

## Quick Start

```bash
git clone https://github.com/Open-Superintelligence-Lab/blueberry-llm-t4-gpu
cd blueberry-llm-t4-gpu
chmod +x setup.sh
./setup.sh
python train.py
```

Try it on [Google Colab](https://colab.research.google.com/drive/1mkiTnK_KaFPX92W5WYCsGq1HaqX7IN4u?usp=sharing)

```bash
# Quick test with 500 steps (fast validation)
# python core/train.py --config dev --max-steps 500
# I think this doesn't work, line above

# Single T4 GPU training (optimized)
python train.py
```

Optimized for **single Tesla T4 GPU** training with native PyTorch implementation. Megatron-LM disabled as it's designed for multi-GPU distributed training.

## 🧪 Running Experiments

**Fork this repo** or use Google Colab to run your experiments:

### Option 1: Fork & Local Setup
```bash
# Fork this repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/blueberry-llm-t4-gpu
cd blueberry-llm-t4-gpu
chmod +x setup.sh
./setup.sh
python train.py
```

### Option 2: Google Colab
Use the [Colab notebook](https://colab.research.google.com/drive/1UE82keuNStPPaeCF50zSgXVHWiywo_pm?usp=sharing) for immediate experimentation.

### Experiment Documentation
When sharing results, include:
- **Clear hypothesis**: What you're testing
- **Setup**: Hardware for this repo will be 1xT4 GPU (Google Colab, Kaggle,...)
- **Results**: Metrics, performance, findings
- **Reproducibility**: How can others reproduce your results?
- **Conclusion**: What you learned

**Good experiment example:**
- Hypothesis: "FP16 vs FP32 training on T4 GPU memory usage"
- Setup: Google Colab T4, 1000 steps, default config
- Results: FP16 used 12GB vs FP32 15GB, 2x faster training
- Reproduce: `python train.py --precision fp16`

## 🖥️ T4 GPU Optimization

Blueberry LLM is specifically optimized for Tesla T4 GPU performance:

| GPU | Memory | Status | Optimization | Notes |
|-----|--------|--------|-------------|-------|
| **Tesla T4** (Google Colab) | 16GB | ✅ **Fully Optimized** | ✅ FP16 Tensor Cores | Max memory utilization (~13-14GB) |
| **Tesla T4** (Cloud/AWS) | 16GB | ✅ **Fully Optimized** | ✅ FP16 Tensor Cores | Single GPU training optimized |

**Note**: This version is optimized specifically for T4 GPUs. Other GPU types may work but are not optimized.

---

This is an **open research project** - we encourage everyone to fork the project, run experiments, and submit pull requests with improvements.

## 📁 Project Structure

```
blueberry-llm/
├── 📁 core/                    # Main functionality
│   ├── train.py               # T4-optimized training
│   ├── inference.py           # Model inference
│   └── t4_config.py           # T4-specific configuration logic
├── 📁 models/                 # Neural network components
├── 📁 data/                   # Data pipeline
├── 📁 optimizers/             # Advanced optimizers
├── 📁 training/               # Training infrastructure
├── 📁 ops/                    # GPU-adaptive operations
├── 📁 system/                 # Hardware detection
├── 📁 configs/                # Configuration management
├── 📁 tests/                  # Testing and examples
├── 📁 docs/                   # Documentation
└── 📁 legacy/                 # Legacy files for reference
```

## 🚀 Usage

### Training
```bash
# Auto-configured training (recommended)
python train.py

# Quick test with 500 steps (fast validation)
python core/train.py --config dev --max-steps 500

# Manual configuration
python core/train.py --config dev
python core/train.py --d-model 768 --n-layers 12
```

### Inference
```bash
# Generate text from trained model
python inference.py "Your prompt here"

# Interactive mode
python inference.py --interactive
```

### Testing
```bash
# Run GPU-adaptive system tests
python test.py

# Run integration examples
python tests/example_integration.py
```

## Research Questions

- Can we achieve better parameter efficiency with sparse expert activation?
- Can we improve expert routing? How does load balancing affect model performance and convergence?

## Engineering Questions
- Should we make it auto detect number and type of GPUs?

## Future Project

Test with [Token Order Prediction](https://github.com/zaydzuhri/token-order-prediction) - "Predicting the Order of Upcoming Tokens Improves Language Modeling"

## Contributing

We welcome contributions! Fork the repo, experiment with different architectures, and submit PRs with your findings.

## Vision

Any company or person (even with no technical experience) should be able to download this repository and run it on their Tesla T4 GPU setup. The system will automatically detect your T4 GPU configuration, tune hyperparameters for optimal T4 performance, and run the best possible training with or without manual configuration from your side.

## 📅 TIMELINE

*Community experiments and findings will be documented here*

### Sep 22 2025
- **Repository Launch**: Initial T4-optimized MoE implementation
- *Your experiment results will appear here when you submit them*

## 📚 Citation

If you use this repository in your research, please cite:

```bibtex
@software{blueberry_llm_t4,
  title={Blueberry LLM: Pretrain LLM On A Single T4 GPU,
  author={Vuk Rosić},
  year={2025},
  url={https://github.com/Open-Superintelligence-Lab/blueberry-llm-t4-gpu},
  note={Tesla T4 GPU-optimized LLM for accessible LLM training}
}
```
