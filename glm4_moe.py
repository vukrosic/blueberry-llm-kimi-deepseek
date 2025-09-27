# Setup, guards, and libs
import math, time, random, threading
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch import amp
import matplotlib.pyplot as plt

try:
    from transformers import Glm4MoeConfig, Glm4MoeForCausalLM
    MODEL_KIND = 'glm4_moe'
except Exception as e:
    print('WARN: Glm4Moe not found, falling back to dense GLM4:', e)
    from transformers import Glm4Config as Glm4MoeConfig, Glm4ForCausalLM as Glm4MoeForCausalLM
    MODEL_KIND = 'glm4_dense_fallback'

try:
    import pynvml
    pynvml.nvmlInit(); _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_AVAILABLE = True
except Exception as e:
    NVML_AVAILABLE = False
    _nvml_handle = None
    print('NVML not available:', e)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
assert DEVICE == 'cuda', 'Requires a CUDA GPU (T4 recommended)'
GPU_NAME = torch.cuda.get_device_name(0)
if 'T4' not in GPU_NAME:
    print(f'NOTE: Intended for NVIDIA T4. Detected: {GPU_NAME}. Proceeding anyway.')

CC_MAJOR, CC_MINOR = torch.cuda.get_device_capability(0)
ALLOW_TF32 = (CC_MAJOR*10+CC_MINOR) >= 80  # Ampere+
torch.backends.cuda.matmul.allow_tf32 = bool(ALLOW_TF32)
torch.backends.cudnn.allow_tf32 = bool(ALLOW_TF32)

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
RUN_DIR = Path('runs')/f"t4_glm4moe_ampfp32_ext_{datetime.now().strftime('%Y%m%d_%H%M%S')}"; RUN_DIR.mkdir(parents=True, exist_ok=True)
print({'gpu': GPU_NAME, 'sm': f'{CC_MAJOR}.{CC_MINOR}', 'tf32': bool(ALLOW_TF32), 'nvml': NVML_AVAILABLE, 'kind': MODEL_KIND})
if not NVML_AVAILABLE:
    print('\nNOTE: NVML power metrics unavailable in this environment. Joules/token will be NaN.\n')

    # Utilities: memory, timing, power sampling
def get_free_total_gb(device: int = 0):
    free, total = torch.cuda.mem_get_info(device)
    return free/(1024**3), total/(1024**3)

class PowerSampler:
    def __init__(self, handle, interval_s=0.01):
        self.handle = handle; self.interval_s = interval_s
        self.samples=[]; self._running=False; self._thr=None
    def _run(self):
        last = time.perf_counter()
        while self._running:
            try:
                watts = pynvml.nvmlDeviceGetPowerUsage(self.handle)/1000.0
            except Exception:
                watts = float('nan')
            now = time.perf_counter(); dt = max(0.0, now-last)
            self.samples.append((now, watts, dt)); last = now
            time.sleep(self.interval_s)
    def start(self):
        if not NVML_AVAILABLE: return
        self.samples.clear(); self._running=True
        self._thr=threading.Thread(target=self._run, daemon=True); self._thr.start()
    def stop(self):
        if not NVML_AVAILABLE: return
        self._running=False
        if self._thr is not None:
            self._thr.join()
    def summary(self):
        if not self.samples: return float('nan'), float('nan'), 0.0
        total_dt=sum(dt for _,_,dt in self.samples)
        energy_j=sum(w*dt for _,w,dt in self.samples)
        avg_w=energy_j/total_dt if total_dt>0 else float('nan')
        return avg_w, energy_j, total_dt

def cuda_event_ms(run_fn):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record(); run_fn(); e.record(); torch.cuda.synchronize();
    return s.elapsed_time(e)

def empty_cache_and_reset():
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


# GLM4 MoE small builder + sizing utils
VOCAB_SIZE = 151552

def _choose_heads(d_model: int, target_head_dim: int = 64) -> int:
    n = max(1, d_model//max(1,target_head_dim)); n=min(n,d_model)
    while n>1 and (d_model % n)!=0: n-=1
    return max(1,n)

def build_glm4_moe(layers: int, d_model: int, n_heads: int, seqlen: int, device='cuda'):
    head_dim = d_model // n_heads
    cfg = Glm4MoeConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=d_model,
        num_hidden_layers=layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        head_dim=head_dim,
        max_position_embeddings=max(2048, seqlen),
        attention_dropout=0.0,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=max(1024, 4*d_model//2),
        n_shared_experts=1,
        use_cache=False,
        tie_word_embeddings=False,
        pad_token_id=151329,
        eos_token_id=[151329],
        attention_bias=True,
    )
    model = Glm4MoeForCausalLM(cfg).to(device); model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, optim

def estimate_memory_gb(layers, d_model, bs, seqlen, precision='fp32'):
    param_count = 4 * layers * (d_model**2)
    param_bytes = param_count * 4
    act_count = 2 * layers * bs * seqlen * d_model
    act_bytes = act_count * (2 if precision=='amp' else 4)
    opt_bytes = param_count * 8
    return (param_bytes+act_bytes+opt_bytes)/(1024**3)

# Train step and experiment runner with OOM backoff
WARMUP_STEPS = 3
REPEATS = 5
DTYPE_AUTOMIXED = torch.float16  # AMP autocast dtype

def train_step(model, optim, batch, precision='fp32', scaler=None, autocast_dtype=torch.float16):
    optim.zero_grad(set_to_none=True)
    if precision == 'amp':
        assert scaler is not None
        with amp.autocast('cuda', dtype=autocast_dtype):
            out = model(input_ids=batch, labels=batch); loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optim); scaler.update()
    else:
        out = model(input_ids=batch, labels=batch); loss = out.loss
        loss.backward(); optim.step()
    return float(loss.detach().item())

def _single_attempt(layers, d_model, n_heads, bs, seqlen, precision, use_ckpt=False, grad_accum=1):
    """
    Returns a result dict or raises torch.cuda.OutOfMemoryError which the caller handles.
    """
    model, optim = build_glm4_moe(layers, d_model, n_heads, seqlen, DEVICE)
    if use_ckpt:
        try: model.gradient_checkpointing_enable()
        except Exception: pass

    scaler = amp.GradScaler('cuda') if precision == 'amp' else None
    batch = torch.randint(0, VOCAB_SIZE, (bs, seqlen), device=DEVICE, dtype=torch.long)

    empty_cache_and_reset()

    # Warmup
    for _ in range(WARMUP_STEPS):
        if grad_accum == 1:
            train_step(model, optim, batch, precision, scaler, DTYPE_AUTOMIXED)
        else:
            optim.zero_grad(set_to_none=True)
            for _k in range(grad_accum):
                if precision == 'amp':
                    with amp.autocast('cuda', dtype=DTYPE_AUTOMIXED):
                        out = model(input_ids=batch, labels=batch); loss = out.loss / grad_accum
                    scaler.scale(loss).backward()
                else:
                    out = model(input_ids=batch, labels=batch); loss = out.loss / grad_accum
                    loss.backward()
            if precision == 'amp':
                scaler.step(optim); scaler.update()
            else:
                optim.step()
        torch.cuda.synchronize()

    # Determine a timing window
    def _one_window():
        if grad_accum == 1:
            train_step(model, optim, batch, precision, scaler, DTYPE_AUTOMIXED)
        else:
            optim.zero_grad(set_to_none=True)
            for _k in range(grad_accum):
                if precision == 'amp':
                    with amp.autocast('cuda', dtype=DTYPE_AUTOMIXED):
                        out = model(input_ids=batch, labels=batch); loss = out.loss / grad_accum
                    scaler.scale(loss).backward()
                else:
                    out = model(input_ids=batch, labels=batch); loss = out.loss / grad_accum
                    loss.backward()
            if precision == 'amp':
                scaler.step(optim); scaler.update()
            else:
                optim.step()

    t_single_ms = cuda_event_ms(_one_window)
    window_steps = 1
    if NVML_AVAILABLE and t_single_ms < 40.0:
        # lengthen short steps to stabilize power integration
        window_steps = min(10, max(2, int(math.ceil(200.0 / max(1.0, t_single_ms)))))

    times_ms, watts_list, energy_list = [], [], []
    mem_peak = 0.0

    for _ in range(REPEATS):
        sampler = PowerSampler(_nvml_handle, 0.01) if NVML_AVAILABLE else None
        if sampler: sampler.start()

        def _window():
            for _i in range(window_steps):
                _one_window()

        ms = cuda_event_ms(_window)

        if sampler:
            sampler.stop()
            avg_w, e_j, _ = sampler.summary()
            watts_list.append(avg_w)
            # Normalize energy by the number of effective steps in the window
            energy_list.append(e_j / window_steps)

        times_ms.append(ms / window_steps)
        mem_peak = max(mem_peak, torch.cuda.max_memory_allocated() / (1024**3))

    mean_ms = float(np.mean(times_ms))
    tokens = bs * seqlen
    tps = tokens / (mean_ms / 1000.0)

    if NVML_AVAILABLE and energy_list:
        avg_w = float(np.mean(watts_list))
        mean_e = float(np.mean(energy_list))
        ept = mean_e / tokens
        tpj = tokens / mean_e if mean_e > 0 else float('nan')
    else:
        avg_w = float('nan'); mean_e = float('nan'); ept = float('nan'); tpj = float('nan')

    return dict(
        layers=layers, d_model=d_model, bs=bs, seqlen=seqlen, precision=precision,
        tps=tps, time_ms=mean_ms, mem_gb=float(mem_peak),
        avg_watts=avg_w, energy_j=mean_e, energy_per_token_j=ept, tokens_per_joule=tpj,
        status='ok',
        adjustments=('ckpt' if use_ckpt else ''),
        grad_accum=grad_accum
    )

def run_experiment(layers, d_model, bs, seqlen, precision='fp32'):
    """
    Strategy (no pre-skip): base -> checkpoint -> grad_accum=2 -> halve bs -> halve seqlen (recursive)
    Always returns a dict with 'status' and 'adjustments' filled in.
    """
    n_heads = _choose_heads(d_model, 64)

    # 1) Base
    try:
        res = _single_attempt(layers, d_model, n_heads, bs, seqlen, precision, use_ckpt=False, grad_accum=1)
        return res
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    # 2) Checkpoint
    try:
        res = _single_attempt(layers, d_model, n_heads, bs, seqlen, precision, use_ckpt=True, grad_accum=1)
        res['adjustments'] = 'ckpt'
        return res
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    # 3) Grad accumulation = 2 (+ checkpoint)
    try:
        res = _single_attempt(layers, d_model, n_heads, bs, seqlen, precision, use_ckpt=True, grad_accum=2)
        res['adjustments'] = 'ckpt,accum2'
        return res
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    # 4) Halve batch, then 5) halve seqlen (recurse)
    new_bs = max(1, bs // 2)
    if new_bs < bs:
        child = run_experiment(layers, d_model, new_bs, seqlen, precision)
        if child['status'] == 'ok':
            child['adjustments'] = (child.get('adjustments', '') + (',' if child.get('adjustments') else '') + 'halve_bs').strip(',')
        return child

    new_sl = max(128, seqlen // 2)
    if new_sl < seqlen:
        child = run_experiment(layers, d_model, bs, new_sl, precision)
        if child['status'] == 'ok':
            child['adjustments'] = (child.get('adjustments', '') + (',' if child.get('adjustments') else '') + 'halve_sl').strip(',')
        return child

    # Give up
    return dict(
        layers=layers, d_model=d_model, bs=bs, seqlen=seqlen, precision=precision,
        tps=float('nan'), time_ms=float('nan'), mem_gb=float('nan'),
        avg_watts=float('nan'), energy_j=float('nan'), energy_per_token_j=float('nan'),
        tokens_per_joule=float('nan'), status='oom', adjustments='', grad_accum=1
    )


# Grid and runner (extended grid) + pre-grid memory sanity print
from itertools import product

GRID = {
    'layers': [4, 8, 12],
    'd_model': [256, 512],
    'batch_size': [1, 2, 4, 8],
    'seqlen': [128, 256, 512, 1024],
    'precision': ['fp32', 'amp'],
}

def iter_grid(grid):
    keys = list(grid.keys()); vals = [grid[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo))

def experiment_key(cfg):
    return f"l{cfg['layers']}_d{cfg['d_model']}_bs{cfg['batch_size']}_sl{cfg['seqlen']}_{cfg['precision']}"

print('Total configs:', len(list(iter_grid(GRID))))

# Memory sanity print (helps diagnose sticky "free" memory)
torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
free, total = get_free_total_gb(0)
print(f"Pre-grid GPU mem: free={free:.2f} GiB | total={total:.2f} GiB | "
      f"allocated={torch.cuda.memory_allocated()/1024**3:.2f} GiB | "
      f"reserved={torch.cuda.memory_reserved()/1024**3:.2f} GiB")

from tqdm.auto import tqdm
import json

results = []
for cfg in tqdm(list(iter_grid(GRID)), desc='Running experiments'):
    r = run_experiment(cfg['layers'], cfg['d_model'], cfg['batch_size'], cfg['seqlen'], cfg['precision'])
    r['key'] = experiment_key(cfg)
    results.append(r)

df = pd.DataFrame(results)
RUN_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(RUN_DIR/'results.csv', index=False)
with open(RUN_DIR/'results.json', 'w') as f:
    json.dump(results, f, indent=2)

df.head(10)
