# Contributing: Ablation Experiments

## Overview

This project trains a GPT-3 Small (125M parameter) decoder-only transformer on FineWeb-Edu and runs ablation studies on individual architectural components. Each teammate owns one ablation variant, modifying only the model architecture while the training infrastructure stays identical across all experiments.

All hyperparameters are anchored to the GPT-3 paper (Brown et al., 2020): d_model=768, 12 heads, 12 layers, batch size ~0.5M tokens, cosine LR schedule with warmup, AdamW with weight decay on matrices only.

## Project Structure

```
Language_Model/
  utils.py                # Shared data pipeline, metrics, plotting
  train.py                # Shared training script (dynamic model loading)
  CONTRIBUTING.md         # This file

  baseline/               # Reference: Sinusoidal PE, Pre-LN, ReLU
    model.py              # Model architecture + CONFIG
    train.sh              # SLURM training job
    runs/<jobid>/         # Output artifacts (auto-created)

  pe_learned/             # Ablation: learned positional embeddings
  pe_rope/                # Ablation: RoPE positional encoding
  pe_alibi/               # Ablation: ALiBi positional encoding
  act_gelu/               # Ablation: GELU activation
  act_swiglu/             # Ablation: SwiGLU activation
  norm_rmsnorm/           # Ablation: RMSNorm (pre-norm position)
  norm_postln/            # Ablation: Post-LayerNorm
```

Each ablation directory has the same layout: `model.py`, `train.sh`.

## Do Not Edit

- **`utils.py`** — shared data pipeline and utilities. Editing this breaks everyone's runs.
- **`train.py`** — shared training loop. If you find a bug, discuss with the team first.
- **Other people's directories** — each person works only in their own ablation folder.

## Baseline Model

All ablations compare against the baseline (sinusoidal PE, pre-LayerNorm, ReLU). The trained baseline checkpoint is shared via Google Drive so everyone has the same reference point. We cant track the trained baseline with git because the weights file is way too big for GitHub.

**Google Drive folder:** [HERE](https://drive.google.com/drive/folders/1vS5C_1KBZ4y2-Cwzf6ELG84r3wbCkH0Z?usp=sharing)

**To set up the baseline checkpoint on your cluster:**

1. Download `baseline.pt` from the Google Drive folder
2. Place it in `baseline/runs/<jobid>/` alongside the other artifacts (which are tracked by git):

You need the baseline checkpoint to run comparison analyses (e.g., plotting baseline loss curves alongside your variant, or computing baseline perplexity at different sequence lengths).

## Interface Contract

Your `model.py` must export exactly three things. The shared `train.py` imports them dynamically at runtime:

```
train.py --model-dir <your_ablation>
    imports <your_ablation>/model.py
        CONFIG            dict of hyperparameters
        attention_net     model class (nn.Module)
        get_pos_encoding  function returning positional encoding
```

### 1. `CONFIG` (dict)

Must include all of these keys:

```python
CONFIG = {
    "model": "your_ablation_name",   # identifies this variant in artifacts
    "bs": 64,                        # batch size per forward pass
    "hidden_size": 768,              # d_model
    "num_heads": 12,
    "num_blocks": 12,
    "dropout": 0.1,
    "seq_length": 1024,
    "tokens_per_epoch": 50_000_000,
    "num_epochs": 100,                # 100 * 50M = 5B tokens
    "peak_lr": 6e-4,
    "accumulation_steps": 8,         # effective batch = 64 * 1024 * 8 = 524K tokens
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_fraction": 0.075,
    "min_lr_fraction": 0.1,          # cosine decay to 10% of peak LR
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
}
```

You may change values if your ablation requires it (e.g., different `num_epochs`) and add new keys for your own use, but **do not remove or rename existing keys**.

### 2. `attention_net` (class)

Top-level model class. Must match this interface exactly:

```python
class attention_net(nn.Module):
    def __init__(self, vocab_size, d, num_heads, num_blocks, seq_length, dropout):
        ...

    def forward(self, word_seq, pos):
        # word_seq: (seq_length, batch_size) — token IDs
        # pos: output of get_pos_encoding(), tensor or None
        # Returns: (seq_length, batch_size, vocab_size) — logits
        ...
```

Do not change constructor or forward signatures. Inside the class, implement whatever architecture you need.

### 3. `get_pos_encoding` (function)

```python
def get_pos_encoding(seq_length, hidden_size, device):
    # Returns: torch.Tensor or None
    # The return value is passed as `pos` to attention_net.forward()
    ...
```

Examples:

```python
# Sinusoidal (baseline) — returns a tensor
def get_pos_encoding(seq_length, hidden_size, device):
    return generate_positional_encoding(seq_length, hidden_size).to(device)

# RoPE — return precomputed frequencies
def get_pos_encoding(seq_length, hidden_size, device):
    return precompute_rope_frequencies(seq_length, hidden_size).to(device)

# ALiBi — return None (bias is computed inside attention)
def get_pos_encoding(seq_length, hidden_size, device):
    return None
```

## Workflow

### Step 1: Implement Your Ablation

Edit `model.py` in your ablation directory. Start from the baseline copy already there and modify the relevant component. Use `baseline/model.py` as your reference implementation.

### Step 2: Train

```bash
cd ~/Language_Model/<your_ablation>
sbatch train.sh
```

This submits a SLURM job on H100-96 GPUs. Training takes approximately 12 hours for 100 epochs.

### Step 3: Monitor

```bash
# Check job status
squeue -u $USER

# Watch training progress (logs appear in your ablation directory)
tail -f logs/<job_name>_<jobid>.out

# Check for errors
tail -f logs/<job_name>_<jobid>.err
```

### Step 4: Check Training Artifacts

After training (or even during — artifacts are saved incrementally each epoch), your `runs/<jobid>/` directory contains:

| File | Contents |
|------|----------|
| `<model_name>.pt` | Best checkpoint (e.g., `baseline.pt`, `pe_rope.pt`). Named after `CONFIG["model"]`. Contains `model_state_dict`, `optimizer_state_dict`, `epoch`, `eval_loss`. |
| `metrics.json` | Full training history: per-epoch `train_ppls`, `eval_ppls`, `tokens_per_sec`, plus `config`, `best_eval_ppl`, `best_epoch`, `peak_memory_gb`. |
| `curves.png` | 3-panel plot: perplexity, log loss, throughput over epochs. |

Artifacts survive SLURM timeouts — they are saved after every epoch, not just at the end.

### Step 5: Collect Analysis Metrics

After training, you must collect ablation-specific metrics for the final report. These require custom evaluation scripts that you write and run against your checkpoint. See the next section for what to measure.

## Analysis Requirements by Ablation Category

### Positional Encoding (baseline, pe_learned, pe_rope, pe_alibi)

**1. Length Generalization**

Train on `seq_length=1024`. Evaluate perplexity at sequence lengths: 1024, 1536, 2048, 3072, 4096. Plot eval perplexity vs. sequence length with one line per variant.

- Sinusoidal: expect noticeable degradation beyond training length
- RoPE: expect moderate degradation
- ALiBi: expect minimal degradation (designed for length extrapolation)
- Learned: **cannot extrapolate** — the embedding table has exactly 1024 entries, so evaluation beyond training length is impossible. Note this in the report; no data needed.

The model architecture is length-agnostic (no hardcoded sequence length in attention or FFN layers). To evaluate at longer lengths, load the checkpoint and call `get_pos_encoding` with the desired length. Reduce eval batch size if needed to fit in GPU memory at longer sequences.

**2. Per-Position Loss Curve**

During evaluation on `seq_length=1024`, compute cross-entropy loss at each token position separately (position 0 through 1023). Plot loss vs. position.

Early positions should have higher loss (less context). The comparison of interest is how each PE handles late positions — ALiBi's recency bias and RoPE's rotational decay create different long-range attention patterns.

The model's forward pass returns `(seq_length, batch_size, vocab_size)`, so per-position loss can be computed directly from the output without any model changes.

**3. Attention Pattern Heatmaps**

For a small eval batch, extract attention weights from each layer and head. Visualize as heatmaps (layer x head grid).

`F.scaled_dot_product_attention` (used in the baseline) does not return attention weights. To extract them, manually compute `softmax(QK^T / sqrt(d_head) + mask)` by hooking into or bypassing the attention module. For ALiBi, include the bias term in the pre-softmax scores.

Expected patterns:
- Sinusoidal/Learned: relatively uniform attention
- RoPE: heads specializing in local vs. global attention
- ALiBi: visible diagonal recency band, width varying by head

### Activation Function (baseline, act_gelu, act_swiglu)

**1. Dead Neuron Count**

After training, pass ~100 eval batches through the model. For each FFN hidden unit, record whether it ever produced a non-zero output (after the activation, before the down-projection). Report dead neuron percentage per layer.

Expected:
- ReLU (baseline): highest dead neuron count — units stuck in negative regime never recover
- GELU: near-zero dead neurons (output is never exactly zero)
- SwiGLU: distinguish between "dead" (gate always near 0) and "selective" (gate active for some inputs)

Plot dead neuron % vs. layer depth. Deeper layers tend to have more dead neurons with ReLU.

**2. Activation Distribution Histograms**

For each variant, collect FFN hidden activations (after the nonlinearity, before the down-projection) across eval batches. Plot the distribution on a log scale.

Expected:
- ReLU: spike at exactly zero plus a positive tail
- GELU: smooth distribution with a slight negative tail
- SwiGLU: the gate-value product creates a distinct distribution — look for sparsity patterns (many values near zero, some large activations)

### Normalization (baseline, norm_rmsnorm, norm_postln)

**1. Training Stability / Loss Curve Comparison**

No custom analysis script needed. After all three variants are trained, plot their loss curves from `metrics.json` on the same axes. This is the primary metric for normalization ablations.

Expected: Post-LN may show loss spikes, slower convergence, or outright divergence compared to Pre-LN and RMSNorm. If it diverges, that is itself a finding worth reporting — include the diverging curve alongside the stable ones.

## Tips

- Read `baseline/model.py` thoroughly before starting. Understand the full architecture before modifying a component.
- Run a short test first if you can: `python train.py --model-dir <your_ablation> --output-dir /tmp/test` — but this is a 125M param model, so local testing may not be feasible.
- If your training run gets killed by SLURM timeout, your artifacts up to the last completed epoch are preserved. Check `metrics.json` to see how far you got, then decide whether to re-run with fewer epochs.
- The GPT-2 BPE tokenizer (`vocab_size=50257`) is shared across all variants via `AutoTokenizer.from_pretrained("gpt2")`. Do not change the tokenizer.
- Training data is streamed from HuggingFace (FineWeb-Edu). No local data files needed.
