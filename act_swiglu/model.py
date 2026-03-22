"""
Baseline decoder-only transformer.
Sinusoidal PE | Pre-LayerNorm | ReLU MLP | GPT-2 BPE tokenizer

This file defines the model architecture and CONFIG for the baseline.
Teammates: edit this file to implement your ablation.
See CONTRIBUTING.md for the interface contract.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_positional_encoding

# ── Config ──────────────────────────────────────────────────────────────────────

CONFIG = {
    "model": "act_swiglu",
    "bs": 64,
    "hidden_size": 768,
    "num_heads": 12,
    "num_blocks": 12,
    "dropout": 0.1,
    "seq_length": 1024,
    "tokens_per_epoch": 50_000_000,
    "num_epochs": 85,              # 85 * 50M = 4.25B tokens (~1.7x Chinchilla optimal for 125M)
    "peak_lr": 6e-4,               # GPT-3 Table 2.1
    "accumulation_steps": 8,        # effective batch = 64 * 1024 * 8 = 524K ≈ GPT-3's 0.5M
    "weight_decay": 0.1,           # GPT-3 Appendix B
    "grad_clip": 1.0,
    "warmup_fraction": 0.075,      # ~750 steps, matches GPT-3's 375M token warmup
    "min_lr_fraction": 0.1,        # cosine decay to 10% of peak (GPT-3)
    "adam_beta1": 0.9,             # GPT-3 Appendix B
    "adam_beta2": 0.95,            # GPT-3 Appendix B (not PyTorch default 0.999)
}

# ── Positional encoding ─────────────────────────────────────────────────────────

def get_pos_encoding(seq_length, hidden_size, device):
    """Return positional encoding tensor for this model variant.
    Baseline uses sinusoidal PE. Override in ablation model.py for RoPE/ALiBi/etc.
    """
    return generate_positional_encoding(seq_length, hidden_size).to(device)

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
