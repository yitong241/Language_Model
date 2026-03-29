"""
Attention heatmap evaluation script for PE ablations.

Loads a trained checkpoint, runs evaluation on the test set, and saves
a normalized attention grid plot.

Usage:
    python eval_attn.py --weights pe_rope/runs/42/pe_rope.pt \
                        --model-py pe_rope/model.py \
                        --output-dir pe_rope/runs/eval
"""

import argparse
import importlib.util
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from utils import load_eval_buffer


# ── Attention utilities ──────────────────────────────────────────────────────

def normalize_attention(attn):
    """
    attn: (seq_len, seq_len) — causal attention weights (lower triangle)
    Returns attention normalized by the uniform baseline per row.
    Each element shows how many times more it is attended to vs. random chance.
    """
    seq_len = attn.shape[0]
    uniform = torch.tensor([1.0 / (i + 1) for i in range(seq_len)],
                            device=attn.device).unsqueeze(1)
    return attn / uniform


def plot_attention_grid(mean_attn_scores, output_dir,
                        use_global_attn=True,
                        min_block=None, max_block=None,
                        min_head=None, max_head=None,
                        cell_size=2.5):
    """
    Plot normalized attention heatmaps in a grid and save to output_dir.

    Args:
        mean_attn_scores: (num_blocks, num_heads, seq_len, seq_len)
        output_dir:       directory to save the figure
        use_global_attn:  if True, all subplots share one colour scale (global halfrange),
                          making cross-head/layer comparisons meaningful.
                          if False, each subplot is independently scaled to its own range,
                          revealing the internal structure of each head.
        min_block, max_block: inclusive block range (default: all)
        min_head,  max_head:  inclusive head range  (default: all)
        cell_size: size in inches of each subplot cell
    """
    num_blocks, num_heads = mean_attn_scores.shape[:2]

    block_start = min_block if min_block is not None else 0
    block_end   = max_block if max_block is not None else num_blocks - 1
    head_start  = min_head  if min_head  is not None else 0
    head_end    = max_head  if max_head  is not None else num_heads - 1

    row_blocks = list(range(block_start, block_end + 1))
    col_heads  = list(range(head_start,  head_end  + 1))

    fs = cell_size * 3.2

    fig, axes = plt.subplots(
        len(row_blocks), len(col_heads),
        figsize=(cell_size * len(col_heads), cell_size * len(row_blocks)),
        squeeze=False,
    )

    all_log_norms = np.empty((num_blocks, num_heads, *mean_attn_scores.shape[2:]))
    for b in range(num_blocks):
        for h in range(num_heads):
            tgt = mean_attn_scores[b, h]
            ln = torch.log(normalize_attention(tgt) + 1e-9)
            ln[tgt == 0] = np.nan
            all_log_norms[b, h] = ln

    global_halfrange = float(np.nanmax(np.abs(all_log_norms)))

    for r, block_idx in enumerate(row_blocks):
        for c, head_idx in enumerate(col_heads):
            ax = axes[r][c]
            log_norm = all_log_norms[block_idx, head_idx]

            if use_global_attn:
                norm = mcolors.CenteredNorm(vcenter=0, halfrange=global_halfrange)
            else:
                norm = mcolors.CenteredNorm(vcenter=0)

            im = ax.imshow(log_norm, cmap='seismic', norm=norm)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=fs * 0.8)

            ax.set_title(f'Block {block_idx}, Head {head_idx}', fontsize=fs)
            ax.set_xlabel('Key pos', fontsize=fs * 0.9)
            ax.set_ylabel('Query pos', fontsize=fs * 0.9)
            ax.tick_params(labelsize=fs * 0.8)

    scale_label = "global scale" if use_global_attn else "per-head scale"
    fig.suptitle(f'Normalized Attention  (log(attn / uniform),  red = preferred)  [{scale_label}]',
                 fontsize=fs * 1.2, y=1.00)
    plt.tight_layout()

    fname = "attn_heatmap_global.png" if use_global_attn else "attn_heatmap_local.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved attention heatmap -> {out_path}")


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_on_test_set_pe(net, pos, eval_data, bs, seq_length, vocab_size, device, criterion):
    for TR_block in net.layer2.decoder.TR_Blocks:
        TR_block.MHA.capture_attn = True

    attn_sum = [None] * len(net.layer2.decoder.TR_Blocks)

    net.eval()
    running_loss = 0
    num_batches  = 0
    for count in range(0, eval_data.size(0) - seq_length, seq_length):
        data  = eval_data[count:count + seq_length].to(device)
        label = eval_data[count + 1:count + seq_length + 1].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            scores = net(data, pos).view(bs * seq_length, vocab_size)
            loss   = criterion(scores, label.view(bs * seq_length))

        for i, block in enumerate(net.layer2.decoder.TR_Blocks):
            attn_weights = block.MHA.attn_weights.float()
            if attn_sum[i] is None:
                attn_sum[i] = attn_weights.sum(axis=0).cpu()
            else:
                attn_sum[i] += attn_weights.sum(axis=0).cpu()

        running_loss += loss.item()
        num_batches  += 1

    attn_mean = [block_sum / (num_batches * bs) for block_sum in attn_sum]
    attn_mean = torch.stack(attn_mean)  # (num_blocks, num_heads, seq_len, seq_len)

    total_loss = running_loss / num_batches
    print(f"  eval ppl = {math.exp(total_loss):.2f}")

    net.train()
    for TR_block in net.layer2.decoder.TR_Blocks:
        TR_block.MHA.capture_attn = False
    return total_loss, attn_mean


# ── Main ─────────────────────────────────────────────────────────────────────

def main(weights_path, model_py_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── Dynamically import model module ──────────────────────────────────────
    # Add the model's directory to sys.path so relative imports inside model.py work
    model_dir = os.path.dirname(os.path.abspath(model_py_path))
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    spec   = importlib.util.spec_from_file_location("model", model_py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    CONFIG           = module.CONFIG
    ModelClass       = module.attention_net
    get_pos_encoding = module.get_pos_encoding

    bs          = CONFIG["bs"]
    hidden_size = CONFIG["hidden_size"]
    num_heads   = CONFIG["num_heads"]
    num_blocks  = CONFIG["num_blocks"]
    dropout     = CONFIG["dropout"]
    seq_length  = CONFIG["seq_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {CONFIG['model']}  (loaded from {model_py_path})")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer    = AutoTokenizer.from_pretrained("gpt2")
    vocab_size   = tokenizer.vocab_size
    eos_token_id = tokenizer.eos_token_id

    # ── Positional encoding ──────────────────────────────────────────────────
    pos = get_pos_encoding(seq_length, hidden_size, device)

    # ── Build model and load weights ─────────────────────────────────────────
    net = ModelClass(vocab_size, hidden_size, num_heads, num_blocks, seq_length, dropout)
    checkpoint = torch.load(weights_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net = net.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}  "
          f"(saved eval ppl = {math.exp(checkpoint['eval_loss']):.2f})")

    # ── Eval buffer ──────────────────────────────────────────────────────────
    print("Loading eval buffer...")
    eval_data = load_eval_buffer(tokenizer, bs, eos_token_id)
    print(f"Eval buffer shape: {eval_data.shape}")

    criterion = nn.CrossEntropyLoss()

    # ── Run evaluation ────────────────────────────────────────────────────────
    _, mean_attn_scores = eval_on_test_set_pe(
        net, pos, eval_data, bs, seq_length, vocab_size, device, criterion
    )

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_attention_grid(mean_attn_scores, output_dir, use_global_attn=True)
    plot_attention_grid(mean_attn_scores, output_dir, use_global_attn=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate attention patterns from a trained checkpoint")
    parser.add_argument("--weights",    type=str, required=True,
                        help="Path to the .pt checkpoint (e.g. pe_rope/runs/42/pe_rope.pt)")
    parser.add_argument("--model-py",   type=str, required=True,
                        help="Path to model.py defining the architecture (e.g. pe_rope/model.py)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save attn_heatmap.png")
    args = parser.parse_args()
    main(args.weights, args.model_py, args.output_dir)
