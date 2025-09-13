# Blueberry LLM 🫐

A Mixture of Experts (MoE) language model implementation.

**Goal: Make LLM training accessible to anyone** - clone, install dependencies, and train your own language model with a single command.

This is an **open research project** - we encourage everyone to fork the project, run experiments, and submit pull requests with improvements.

## Quick Start

```bash
git clone https://github.com/your-username/blueberry
cd blueberry
pip install -r requirements.txt
python llm.py
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
