# Blueberry LLM 🫐

A Mixture of Experts (MoE) language model implementation.

**Goal: Make LLM training accessible to anyone** - clone, install dependencies, and train your own language model with a single command.

## Quick Start

```bash
git clone https://github.com/Open-Superintelligence-Lab/blueberry-llm
cd blueberry-llm
chmod +x setup.sh
./setup.sh
python train_auto.py
```

This is an **open research project** - we encourage everyone to fork the project, run experiments, and submit pull requests with improvements.

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

Any company or person (even with no technical experience) should be able to download this repository and run it on their GPU setup - from 1 GPU to 1 million GPUs. The system will be able to automatically detects your hardware configuration, tunes hyperparameters for optimal performance, and runs the best possible training with or without manual configuration from your side.
