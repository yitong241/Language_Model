# Contributing: Ablation Experiments

## Project Structure

```
Language_Model/
  utils.py              # Shared data pipeline, metrics, plotting — DO NOT EDIT
  train.py              # Shared training script — DO NOT EDIT
  eval.py               # Shared evaluation/generation script — DO NOT EDIT
  baseline/             # Baseline model (reference implementation)
    model.py            # Model architecture + config
    train.sh            # SLURM training job
    eval.sh             # SLURM evaluation job
    runs/<jobid>/       # Output artifacts (auto-created per job)
  pe_learned/           # Ablation: learned positional encoding
  pe_rope/              # Ablation: RoPE positional encoding
  pe_alibi/             # Ablation: ALiBi positional encoding
  act_gelu/             # Ablation: GELU activation
  act_swiglu/           # Ablation: SwiGLU activation
  norm_rmsnorm/         # Ablation: RMSNorm (pre-norm)
  norm_postln/          # Ablation: Post-LayerNorm
```

Each ablation directory has the same structure: `model.py`, `train.sh`, `eval.sh`, and a `runs/` folder for artifacts.

## How It Works

The shared `train.py` dynamically loads your `model.py` at runtime:

```
train.py --model-dir <your_ablation>
    └── imports <your_ablation>/model.py
        ├── CONFIG        (hyperparameters)
        ├── attention_net  (model class)
        └── get_pos_encoding (positional encoding function)
```

This guarantees that the training loop, optimizer, data pipeline, and evaluation are **identical** across all ablation runs. The only thing that differs is the model architecture.

## What You Edit

**Your `model.py`** — this is the only file you need to modify. It must export three things:

### 1. `CONFIG` (dict)

All hyperparameters for your run. Must include these keys:

```python
CONFIG = {
    "model": "your_ablation_name",   # identifies this run in artifacts
    "bs": 64,                        # batch size per forward pass
    "hidden_size": 768,              # model dimension (d_model)
    "num_heads": 12,                 # number of attention heads
    "num_blocks": 12,                # number of transformer blocks
    "dropout": 0.1,
    "seq_length": 1024,              # context window
    "tokens_per_epoch": 50_000_000,  # tokens streamed per epoch
    "num_epochs": 85,                # total epochs
    "peak_lr": 6e-4,                 # peak learning rate
    "accumulation_steps": 8,         # gradient accumulation steps
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_fraction": 0.075,        # fraction of total steps for LR warmup
    "min_lr_fraction": 0.1,          # min LR as fraction of peak
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
}
```

You can change values (e.g., adjust `num_epochs` if your ablation needs more/fewer) and add new keys for your own use, but **do not remove or rename existing keys**.

### 2. `attention_net` (class)

The top-level model class. Must follow this interface:

```python
class attention_net(nn.Module):
    def __init__(self, vocab_size, d, num_heads, num_blocks, seq_length, dropout):
        # Build your model here
        ...

    def forward(self, word_seq, pos):
        # word_seq: (seq_length, batch_size) — token IDs
        # pos: output of get_pos_encoding() — tensor or None
        # Returns: (seq_length, batch_size, vocab_size) — logits
        ...
```

**Do not change the constructor or forward signatures.** The shared training script calls these with exactly these arguments.

Inside the class, you can do whatever you want — change attention mechanisms, normalization, activation functions, weight initialization, etc.

### 3. `get_pos_encoding` (function)

Controls how positional information is provided to the model:

```python
def get_pos_encoding(seq_length, hidden_size, device):
    """Return positional encoding for this model variant.

    Returns:
        torch.Tensor or None — passed as `pos` to attention_net.forward()
    """
    ...
```

**Examples by ablation type:**

```python
# Baseline / learned PE — return a tensor
def get_pos_encoding(seq_length, hidden_size, device):
    return generate_positional_encoding(seq_length, hidden_size).to(device)

# RoPE — return precomputed frequency tensors, or None
def get_pos_encoding(seq_length, hidden_size, device):
    return precompute_rope_frequencies(seq_length, hidden_size).to(device)

# ALiBi — return None (bias computed inside attention)
def get_pos_encoding(seq_length, hidden_size, device):
    return None
```

The return value is passed directly as the `pos` argument to `attention_net.forward()`. Your model decides how to use it (or ignore it).

## How to Submit Jobs

### Training

From your ablation directory:

```bash
cd ~/Language_Model/pe_rope
sbatch train.sh
```

This creates `runs/<jobid>/` containing:
- `best_model.pt` — best checkpoint (by eval perplexity)
- `metrics.json` — per-epoch train/eval PPL, throughput, config
- `curves.png` — training curves plot

Artifacts are saved incrementally after each epoch, so they survive SLURM timeouts.

### Monitoring

```bash
# Check job status
squeue -u $USER

# Watch training progress
tail -f logs/<job_name>_<jobid>.out

# Check for errors
tail -f logs/<job_name>_<jobid>.err
```

### Evaluation (text generation)

From your ablation directory:

```bash
cd ~/Language_Model/pe_rope
sbatch eval.sh runs/<jobid>/best_model.pt
sbatch eval.sh runs/<jobid>/best_model.pt "Custom prompt here"
```

Output goes to `logs/eval_<jobid>.out`.

## What NOT to Edit

- **`utils.py`** — shared data pipeline and utilities. Editing this breaks everyone's runs.
- **`train.py`** (root) — shared training loop. If you find a bug, discuss with the team first.
- **`eval.py`** (root) — shared evaluation script.
- **Other people's directories** — each person works only in their own ablation folder.

## Quick Start: Creating Your Ablation

1. Your directory already exists with a copy of `model.py`
2. Edit `model.py` to implement your ablation
3. Test locally if possible: `python train.py --model-dir your_ablation --output-dir /tmp/test`, but this is a 125M param model. Unlikely you can test locally.
4. Submit: `cd your_ablation && sbatch train.sh`
5. Check results: `cat logs/your_ablation_<jobid>.out`

## Output Artifacts

After training, `runs/<jobid>/` contains:

| File | Description |
|------|-------------|
| `best_model.pt` | Best checkpoint (lowest eval PPL). Contains `model_state_dict`, `optimizer_state_dict`, `epoch`, `eval_loss`. |
| `metrics.json` | Full training history: per-epoch `train_ppls`, `eval_ppls`, `tokens_per_sec`, `peak_memory_gb`, `config`, `best_eval_ppl`, `best_epoch`. |
| `curves.png` | 3-panel plot: Perplexity, Log Loss, Throughput over epochs. |
