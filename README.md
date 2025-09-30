# Blueberry LLM: DeepSeek & GLM4 MoE Research

A comprehensive research framework for implementing and experimenting with advanced transformer architectures, focusing on Mixture of Experts (MoE) models inspired by DeepSeek and GLM4 architectures.

## 🎯 Project Overview

This project provides a modular framework for:
- **Mixture of Experts (MoE) Models**: Implementation of GLM4-style MoE with expert routing
- **DeepSeek Architecture**: Advanced attention mechanisms and MLP designs
- **Hybrid Architectures**: Combining DeepSeek attention with GLM4 MoE
- **Experimental Framework**: Systematic ablation studies and benchmarking
- **Educational Content**: Comprehensive course materials on transformer fundamentals

## 🏗️ Architecture Highlights

### Core Models
- **MoE Minimal LLM**: Efficient MoE implementation with configurable experts
- **DeepSeek V3**: Advanced attention with LoRA-style projections and RoPE scaling
- **GLM4 MoE**: Expert routing with group-based selection and load balancing
- **Hybrid Architectures**: DeepSeek attention + GLM4 MoE combinations

### Key Features
- **Expert Routing**: Top-k expert selection with load balancing
- **Advanced Attention**: Multi-head attention with RoPE positional encoding
- **Flexible Scaling**: Configurable model dimensions and expert counts
- **Efficient Training**: Gradient accumulation, mixed precision, and optimization
- **Comprehensive Evaluation**: HellaSwag benchmarking and custom metrics

## 📁 Project Structure

```
blueberry-llm-kimi-deepseek/
├── _course/                          # Educational materials
│   ├── 01_python_beginner_lessons/   # Python fundamentals
│   ├── 02_math_not_scary/            # Mathematical foundations
│   ├── 03_pytorch_fundamentals/      # PyTorch basics
│   ├── 04_neuron_from_scratch/       # Neural network basics
│   ├── 05_activation_functions/      # Activation functions
│   ├── 06_neural_network_from_scratch/ # Neural networks
│   ├── 07_attention_mechanism/       # Attention mechanisms
│   ├── 08_transformer_feedforward/   # Transformer FF layers
│   ├── 09_building_a_transformer/    # Full transformer
│   ├── 10_deepseek_latent_attention/ # DeepSeek attention
│   └── 11_glm4_moe/                  # GLM4 MoE architecture
├── configs/                          # Configuration files
│   └── moe_config.py                 # MoE model configuration
├── data/                             # Data handling
│   ├── dataset.py                    # Dataset classes
│   └── loader.py                     # Data loading utilities
├── models/                           # Model implementations
│   ├── components.py                 # Model components
│   ├── layers.py                     # Custom layers
│   └── moe_llm.py                    # MoE LLM implementation
├── training/                         # Training utilities
│   ├── trainer.py                    # Training loop
│   └── evaluation.py                 # Evaluation metrics
├── optimizers/                       # Custom optimizers
│   └── muon.py                       # Muon optimizer
├── experiments/                      # Experimental studies
│   ├── exp1_simplified_ablation_study/ # Ablation studies
│   ├── exp2_deepseek_attn_mlp_lr_search/ # Learning rate search
│   └── exp3_deepseek_attn_glm4_moe_lr_expert_search/ # Expert search
├── utils/                            # Utility functions
│   └── helpers.py                    # Helper functions
├── train_moe.py                      # Main training script
├── deepseek_modeling.py              # DeepSeek V3 implementation
└── configuration_deepseek.py         # DeepSeek configuration
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd blueberry-llm-kimi-deepseek
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Basic Usage

#### Train a MoE Model
```bash
python train_moe.py
```

#### Run Experiments
```bash
# Simplified ablation study
cd experiments/exp1_simplified_ablation_study
python exp1_trainer.py

# Learning rate search
cd experiments/exp2_deepseek_attn_mlp_lr_search
python lr_search.py

# Expert configuration search
cd experiments/exp3_deepseek_attn_glm4_moe_lr_expert_search
python expert_search.py
```

## 🔧 Configuration

### MoE Model Configuration

The `MoEModelConfig` class provides extensive configuration options:

```python
@dataclass
class MoEModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    
    # MoE specific
    num_experts: int = 8
    expert_top_k: int = 2
    load_balancing_weight: float = 0.01
    
    # Training
    batch_size: int = 24
    max_steps: int = 20
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01
    
    # Data
    max_seq_len: int = 512
    max_tokens: int = 500000
```

### DeepSeek Configuration

The DeepSeek implementation supports advanced features:
- **RoPE Scaling**: Linear, Dynamic NTK, and Yarn scaling
- **LoRA-style Projections**: Efficient query/key/value projections
- **Flash Attention**: Optional flash attention integration
- **MoE Integration**: DeepSeek attention with MoE feedforward

## 🧪 Experiments

### Experiment 1: Simplified Ablation Study
- **Models**: 5 variants (baseline, MLP, attention+MLP, MoE, attention+MoE)
- **Scale**: 512 hidden dimensions
- **Evaluation**: HellaSwag benchmark integration
- **Focus**: Architecture comparison at target scale

### Experiment 2: Learning Rate Search
- **Focus**: DeepSeek attention + MLP combinations
- **Method**: Systematic learning rate exploration
- **Metrics**: Validation loss, accuracy, perplexity
- **Output**: Optimal learning rate recommendations

### Experiment 3: Expert Configuration Search
- **Focus**: DeepSeek attention + GLM4 MoE
- **Variables**: Expert count, learning rates, top-k values
- **Method**: Grid search with validation
- **Output**: Optimal expert configurations

## 📊 Key Features

### Mixture of Experts (MoE)
- **Expert Routing**: Top-k expert selection with load balancing
- **Load Balancing**: Auxiliary loss for expert utilization
- **Scalable Design**: Configurable expert count and routing
- **Efficient Inference**: Expert selection optimization

### Advanced Attention
- **Multi-Head Attention**: Configurable attention heads
- **RoPE Encoding**: Rotary positional embeddings
- **Flash Attention**: Optional memory-efficient attention
- **LoRA Projections**: Efficient query/key/value transformations

### Training Infrastructure
- **Mixed Precision**: Automatic mixed precision training
- **Gradient Accumulation**: Efficient large batch training
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Optimization**: Muon optimizer with AdamW hybrid

### Evaluation
- **HellaSwag Benchmark**: Standardized evaluation
- **Custom Metrics**: Perplexity, accuracy, expert utilization
- **Visualization**: Training curves and expert analysis
- **Comprehensive Logging**: Detailed training and evaluation logs

## 🎓 Educational Content

The `_course/` directory contains comprehensive educational materials:

1. **Python Fundamentals**: Basic Python programming
2. **Mathematical Foundations**: Linear algebra, calculus, gradients
3. **PyTorch Basics**: Tensors, operations, autograd
4. **Neural Networks**: From neurons to networks
5. **Activation Functions**: ReLU, Sigmoid, Tanh, SiLU, SwiGLU
6. **Attention Mechanisms**: Self-attention, multi-head attention
7. **Transformer Architecture**: Complete transformer implementation
8. **DeepSeek Attention**: Advanced attention mechanisms
9. **GLM4 MoE**: Mixture of experts implementation

## 🔬 Research Applications

This framework is designed for:
- **Architecture Research**: Comparing different transformer variants
- **MoE Studies**: Expert routing and load balancing research
- **Scaling Studies**: Model size and expert count optimization
- **Educational Use**: Learning transformer fundamentals
- **Benchmarking**: Standardized evaluation across architectures

## 📈 Performance

### Model Efficiency
- **Parameter Efficiency**: MoE models activate only a subset of parameters
- **Memory Optimization**: Efficient expert routing and caching
- **Training Speed**: Optimized training loops with mixed precision

### Evaluation Metrics
- **Perplexity**: Language modeling performance
- **Accuracy**: Next-token prediction accuracy
- **Expert Utilization**: Load balancing effectiveness
- **HellaSwag Score**: Standardized benchmark performance

## 🤝 Contributing

We welcome contributions! Please see the following areas:
- **New Architectures**: Additional transformer variants
- **Optimization**: Training and inference optimizations
- **Experiments**: New experimental studies
- **Documentation**: Educational content and examples
- **Evaluation**: Additional benchmarks and metrics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek**: For the advanced attention architecture
- **GLM4**: For the MoE implementation inspiration
- **HuggingFace**: For the transformer library foundation
- **PyTorch**: For the deep learning framework

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact the maintainers.

---

**Happy Training! 🚀**
