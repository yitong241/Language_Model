"""
Shared training script for all ablation experiments.
Dynamically loads model architecture and config from <model-dir>/model.py.

Usage:
    python train.py --model-dir baseline --output-dir baseline/runs/test
    python train.py --model-dir pe_rope --output-dir pe_rope/runs/test
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from utils import (
    display_num_param,
    get_epoch_batches,
    load_eval_buffer,
    save_metrics,
    save_plots,
)


def load_model_module(model_dir):
    """Dynamically import model.py from the given ablation directory."""
    # Ensure project root is on sys.path so model.py can import utils
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    model_path = os.path.join(model_dir, "model.py")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No model.py found in {model_dir}")

    spec = importlib.util.spec_from_file_location("model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model module ────────────────────────────────────────────────────
    model_module = load_model_module(model_dir)
    CONFIG = model_module.CONFIG
    ModelClass = model_module.attention_net
    get_pos_encoding = model_module.get_pos_encoding

    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Model: {CONFIG['model']}")
    print(f"Config: {json.dumps(CONFIG, indent=2)}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    eos_token_id = tokenizer.eos_token_id

    # ── Unpack config ────────────────────────────────────────────────────────
    bs = CONFIG["bs"]
    hidden_size = CONFIG["hidden_size"]
    num_heads = CONFIG["num_heads"]
    num_blocks = CONFIG["num_blocks"]
    dropout = CONFIG["dropout"]
    seq_length = CONFIG["seq_length"]
    tokens_per_epoch = CONFIG["tokens_per_epoch"]
    num_epochs = CONFIG["num_epochs"]
    peak_lr = CONFIG["peak_lr"]
    accumulation_steps = CONFIG["accumulation_steps"]
    weight_decay = CONFIG["weight_decay"]
    grad_clip = CONFIG["grad_clip"]
    warmup_fraction = CONFIG["warmup_fraction"]
    min_lr_fraction = CONFIG["min_lr_fraction"]
    min_lr = peak_lr * min_lr_fraction
    adam_beta1 = CONFIG["adam_beta1"]
    adam_beta2 = CONFIG["adam_beta2"]

    pos = get_pos_encoding(seq_length, hidden_size, device)

    # ── Model ────────────────────────────────────────────────────────────────
    net = ModelClass(vocab_size, hidden_size, num_heads, num_blocks, seq_length, dropout)
    display_num_param(net)
    net = net.to(device)

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # GPT-3 convention: weight decay on matrices only, not biases/LayerNorm
    decay_params = [p for p in net.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in net.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=peak_lr, betas=(adam_beta1, adam_beta2))

    total_steps = num_epochs * (tokens_per_epoch // (bs * seq_length)) // accumulation_steps
    warmup_steps = int(total_steps * warmup_fraction)

    def calc_multiplier(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr / peak_lr + (1 - min_lr / peak_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=calc_multiplier)

    # ── Eval buffer ──────────────────────────────────────────────────────────
    print("Loading eval buffer...")
    eval_data = load_eval_buffer(tokenizer, bs, eos_token_id)
    print(f"Eval buffer shape: {eval_data.shape}")

    # ── Eval function ────────────────────────────────────────────────────────
    @torch.no_grad()
    def eval_on_test_set():
        net.eval()
        running_loss = 0
        num_batches = 0
        num_steps = eval_data.size(0) - seq_length
        for count in range(0, num_steps, seq_length):
            minibatch_data = eval_data[count:count + seq_length]
            minibatch_label = eval_data[count + 1:count + seq_length + 1]
            minibatch_data = minibatch_data.to(device)
            minibatch_label = minibatch_label.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = net(minibatch_data, pos)
                minibatch_label = minibatch_label.view(bs * seq_length)
                scores = scores.view(bs * seq_length, vocab_size)
                loss = criterion(scores, minibatch_label)
            running_loss += loss.item()
            num_batches += 1
        total_loss = running_loss / num_batches
        print(f"  eval ppl = {math.exp(total_loss):.2f}")
        net.train()
        return total_loss

    # ── Training loop ────────────────────────────────────────────────────────
    train_ppls = []
    eval_ppls = []
    tokens_per_sec = []

    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    best_loss = float('inf')
    prev_elapsed = 0

    print(f"\nStarting training: {num_epochs} epochs, {tokens_per_epoch:,} tokens/epoch")
    print(f"Total optimizer steps: {total_steps}, warmup: {warmup_steps}")

    start = time.time()
    for epoch in range(num_epochs):
        running_loss = 0
        num_batches = 0
        step = 0
        net.train()
        optimizer.zero_grad()

        for minibatch_data, minibatch_label in get_epoch_batches(
                tokens_per_epoch, bs, seq_length, tokenizer, eos_token_id, seed=epoch):

            minibatch_data = minibatch_data.to(device)
            minibatch_label = minibatch_label.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                scores = net(minibatch_data, pos)
                scores = scores.view(bs * seq_length, vocab_size)
                minibatch_label = minibatch_label.view(bs * seq_length)
                loss = criterion(scores, minibatch_label)
                loss = loss / accumulation_steps

            loss.backward()
            running_loss += loss.item() * accumulation_steps
            num_batches += 1
            step += 1

            if step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # flush leftover accumulated gradients
        if step % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss = running_loss / num_batches
        elapsed = time.time() - start
        epoch_tokens = num_batches * bs * seq_length
        epoch_time = elapsed - prev_elapsed
        tps = epoch_tokens / epoch_time
        prev_elapsed = elapsed

        current_lr = optimizer.param_groups[0]['lr']
        print(f"epoch={epoch}\t time={elapsed:.1f}s\t lr={current_lr:.6f}\t"
              f" train_ppl={math.exp(total_loss):.2f}\t tok/s={tps:.0f}")

        eval_loss = eval_on_test_set()

        train_ppls.append(math.exp(total_loss))
        eval_ppls.append(math.exp(eval_loss))
        tokens_per_sec.append(tps)

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_loss': eval_loss,
            }, checkpoint_path)
            print(f"  -> saved checkpoint (eval ppl = {math.exp(eval_loss):.2f})")

        # save artifacts incrementally so they survive SLURM timeouts
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        save_metrics(output_dir, CONFIG, train_ppls, eval_ppls, tokens_per_sec, peak_mem)
        save_plots(output_dir, train_ppls, eval_ppls, tokens_per_sec)

    print(f"\nTraining complete. Peak GPU memory: {peak_mem:.2f} GB")
    print(f"Best eval ppl: {math.exp(best_loss):.2f} (epoch {eval_ppls.index(min(eval_ppls))})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer on FineWeb-Edu")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Ablation directory containing model.py (e.g., 'baseline', 'pe_rope')")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for checkpoints, metrics, and plots")
    args = parser.parse_args()
    main(args.model_dir, args.output_dir)
