"""
Baseline decoder-only transformer trained on FineWeb-Edu.
Sinusoidal PE | Pre-LayerNorm | ReLU MLP | GPT-2 BPE tokenizer
"""

import argparse
import math
import os
import sys
import time

# Add parent directory to path so we can import shared utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from utils import (
    generate_positional_encoding,
    display_num_param,
    get_epoch_batches,
    load_eval_buffer,
    save_metrics,
    save_plots,
)

# ── Config ──────────────────────────────────────────────────────────────────────

CONFIG = {
    "model": "baseline",
    "bs": 32,
    "hidden_size": 384,
    "num_heads": 6,
    "num_blocks": 6,
    "dropout": 0.1,
    "seq_length": 512,
    "tokens_per_epoch": 2_000_000,
    "num_epochs": 20,
    "peak_lr": 3e-4,
    "accumulation_steps": 4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "warmup_fraction": 0.05,
    "min_lr_fraction": 0.1,
}

# ── Model ───────────────────────────────────────────────────────────────────────

class MultipleAttentionHead(nn.Module):
    def __init__(self, d, num_heads, dropout):
        super().__init__()
        d_head = d // num_heads
        assert d == d_head * num_heads
        self.WQ = nn.Linear(d, d, bias=False)
        self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False)
        self.WO = nn.Linear(d, d)
        self.dropout_p = dropout
        self.num_heads = num_heads
        self.d_head = d_head

    def forward(self, H):
        batch, seq_len, _ = H.shape
        Q = self.WQ(H).reshape(batch, seq_len, self.num_heads, self.d_head)
        K = self.WK(H).reshape(batch, seq_len, self.num_heads, self.d_head)
        V = self.WV(H).reshape(batch, seq_len, self.num_heads, self.d_head)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(batch, seq_len, self.d_head * self.num_heads)
        H = self.WO(attn_out)
        return H


class TransformerBlock(nn.Module):
    def __init__(self, d, num_heads, dropout):
        super().__init__()
        self.LN_MHA = nn.LayerNorm(d)
        self.LN_MLP = nn.LayerNorm(d)
        self.MHA = MultipleAttentionHead(d, num_heads, dropout)
        self.MLP = nn.Sequential(
            nn.Linear(d, 4 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * d, d),
        )
        self.drop_attn = nn.Dropout(dropout)
        self.drop_mlp = nn.Dropout(dropout)

    def forward(self, H):
        H = H + self.drop_attn(self.MHA(self.LN_MHA(H)))
        H = H + self.drop_mlp(self.MLP(self.LN_MLP(H)))
        return H


class Transformer_decoder(nn.Module):
    def __init__(self, d, num_heads, num_blocks, seq_length, dropout):
        super().__init__()
        self.TR_Blocks = nn.ModuleList(
            [TransformerBlock(d, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(d)

    def forward(self, batch_seq, pos_enc):
        H = batch_seq.transpose(1, 0)
        pos_enc = pos_enc.unsqueeze(dim=0)
        H = H + pos_enc
        for TR_Block in self.TR_Blocks:
            H = TR_Block(H)
        H = self.final_norm(H)
        H = H.permute(1, 0, 2)
        return H


class ANN(nn.Module):
    def __init__(self, d, num_heads, num_blocks, seq_length, dropout):
        super().__init__()
        self.decoder = Transformer_decoder(d, num_heads, num_blocks, seq_length, dropout)

    def forward(self, g_seq, pos):
        return self.decoder(g_seq, pos)


class attention_net(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def __init__(self, vocab_size, d, num_heads, num_blocks, seq_length, dropout):
        super().__init__()
        self.layer1 = nn.Embedding(vocab_size, d)
        self.layer2 = ANN(d, num_heads, num_blocks, seq_length, dropout)
        self.layer3 = nn.Linear(d, vocab_size, bias=False)
        self.layer3.weight = self.layer1.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * num_blocks)
        for block in self.layer2.decoder.TR_Blocks:
            nn.init.normal_(block.MHA.WO.weight, mean=0, std=residual_std)
            nn.init.normal_(block.MLP[3].weight, mean=0, std=residual_std)

    def forward(self, word_seq, pos):
        g_seq = self.layer1(word_seq)
        h_seq = self.layer2(g_seq, pos)
        score_seq = self.layer3(h_seq)
        return score_seq

# ── Training ────────────────────────────────────────────────────────────────────

def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    eos_token_id = tokenizer.eos_token_id

    # ── Unpack config ───────────────────────────────────────────────────────
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

    pos = generate_positional_encoding(seq_length, hidden_size).to(device)

    # ── Model ───────────────────────────────────────────────────────────────
    net = attention_net(vocab_size, hidden_size, num_heads, num_blocks, seq_length, dropout)
    display_num_param(net)
    net = net.to(device)

    # ── Optimizer / Scheduler ───────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=peak_lr, weight_decay=weight_decay)

    total_steps = num_epochs * (tokens_per_epoch // (bs * seq_length)) // accumulation_steps
    warmup_steps = int(total_steps * warmup_fraction)

    def calc_multiplier(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr / peak_lr + (1 - min_lr / peak_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=calc_multiplier)

    # ── Eval buffer ─────────────────────────────────────────────────────────
    print("Loading eval buffer...")
    eval_data = load_eval_buffer(tokenizer, bs, eos_token_id)
    print(f"Eval buffer shape: {eval_data.shape}")

    # ── Eval function ───────────────────────────────────────────────────────
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

    # ── Training loop ───────────────────────────────────────────────────────
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

    # ── Save artifacts ──────────────────────────────────────────────────────
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"\nTraining complete. Peak GPU memory: {peak_mem:.2f} GB")

    save_metrics(output_dir, CONFIG, train_ppls, eval_ppls, tokens_per_sec, peak_mem)
    save_plots(output_dir, train_ppls, eval_ppls, tokens_per_sec)

    print(f"Best eval ppl: {math.exp(best_loss):.2f} (epoch {eval_ppls.index(min(eval_ppls))})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline transformer on FineWeb-Edu")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for checkpoints, metrics, and plots")
    args = parser.parse_args()
    main(args.output_dir)
