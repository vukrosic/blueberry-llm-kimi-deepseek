# T4 Speedrun Challenge 🏃‍♂️

**Goal**: Achieve the lowest validation loss possible in 5 minutes on a Tesla T4 GPU.

## 🎯 Rules

- **Time Limit**: 5 minutes
- **Hardware**: Tesla T4 (16GB VRAM)
- **Dataset**: Fixed dataset (100K tokens, 1K documents)
- **Success Metric**: Validation loss (lower is better)

## 🚀 How to Compete

1. **Copy template**: `cp -r speedrun/submissions/template-change-this speedrun/submissions/your-name`
2. **Modify your folder**: Change architecture, training, optimizations
3. **Run speedrun**: `python speedrun/submissions/your-name/speedrun.py`
4. **Submit PR**: Add your folder to `speedrun/submissions/`

## 📁 Structure

```
speedrun/
├── submissions/
│   ├── template-change-this/    # Start here
│   ├── alice-fast/              # Alice's submission
│   ├── bob-memory/              # Bob's submission
│   └── charlie-moe/             # Charlie's submission
└── README.md
```

## 🎨 Customize Everything

Modify any file in your submission folder:

- `config.py` - Change model architecture, hyperparameters
- `speedrun.py` - Modify training loop, optimizations
- `leaderboard.py` - Custom scoring, metrics

### How to Change Architecture

**Option 1: Quick changes**
```bash
python speedrun/submissions/your-name/speedrun.py --custom-config --d-model 512 --n-layers 8
```

**Option 2: Edit `config.py`**
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

**Option 3: Edit `speedrun.py`**
- Modify the training loop
- Change optimizers, schedulers
- Add custom evaluation logic

## 📊 Leaderboard

```bash
# View leaderboard
python speedrun/submissions/your-name/leaderboard.py --show

# Add your results
python speedrun/submissions/your-name/leaderboard.py --add results.json --participant "YourName"
```

## 🏆 Tips

- **5 minutes is short** - focus on fast convergence
- **T4 has 16GB** - balance model size vs. batch size
- **Validation loss** - that's your score (lower = better)
- **Share your approach** - help others learn and improve

---

**Go optimize!** 🚀
