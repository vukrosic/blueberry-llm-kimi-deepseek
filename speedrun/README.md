# T4 Speedrun Challenge 🏃‍♂️

**Goal**: Achieve the lowest validation loss possible in 5 minutes on a Tesla T4 GPU.

## 🎯 Rules

- **Time Limit**: 5 minutes
- **Hardware**: Tesla T4 (16GB VRAM)
- **Dataset**: Fixed dataset (100K tokens, 1K documents)
- **Success Metric**: Validation loss (lower is better)

## ✅ What You Can Change

- **Everything** in the `speedrun/` directory
- Model architecture, training hyperparameters, optimizations
- Any code, configurations, or strategies you want

## 🚀 Quick Start

```bash
# Run speedrun
python speedrun/speedrun.py

# Custom time limit
python speedrun/speedrun.py --time-limit 3
```

## 📊 Leaderboard

```bash
# View leaderboard
python speedrun/leaderboard.py --show

# Add your results
python speedrun/leaderboard.py --add results.json --participant "YourName"
```

## 🎨 Customize Everything

Modify any file in `speedrun/` directory:

- `config.py` - Change model architecture, hyperparameters
- `speedrun.py` - Modify training loop, optimizations
- `leaderboard.py` - Custom scoring, metrics

### How to Change Architecture

**Option 1: Quick changes**
```bash
python speedrun/speedrun.py --custom-config --d-model 512 --n-layers 8 --batch-size 32
```

**Option 2: Edit `speedrun/config.py`**
```python
def get_my_config():
    return create_custom_t4_config(
        d_model=512,      # Model size
        n_layers=8,       # Depth
        n_heads=8,        # Attention heads
        batch_size=32,    # Batch size
        num_experts=16,   # MoE experts
        muon_lr=0.005,   # Learning rate
    )
```

**Option 3: Edit `speedrun/speedrun.py`**
- Modify the training loop
- Change optimizers, schedulers
- Add custom evaluation logic

## 🏆 Tips

- **5 minutes is short** - focus on fast convergence
- **T4 has 16GB** - balance model size vs. batch size
- **Validation loss** - that's your score (lower = better)

---

**Go optimize!** 🚀
