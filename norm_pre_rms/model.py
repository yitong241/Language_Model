"""
Normalization ablation transformer.
Supports:
1) pre_ln   : Pre-norm with LayerNorm
2) pre_rms  : Pre-norm with RMSNorm
3) post_ln  : Post-norm with LayerNorm

Keep train.py unchanged.
Create 3 folders, each with this file, and only change VARIANT / CONFIG["model"].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_positional_encoding

# ============================================================
# Choose variant here
# ============================================================
VARIANT = "pre_rms"   # one of: "pre_ln", "pre_rms", "post_ln"

CONFIG = {
    "model": f"norm_{VARIANT}",
    "bs": 8,
    "hidden_size": 768,
    "num_heads": 12,
    "num_blocks": 12,
    "dropout": 0.1,
    "seq_length": 1024,
    "tokens_per_epoch": 50_000_000,
    "num_epochs": 85,
    "peak_lr": 6e-4,
    "accumulation_steps": 8,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_fraction": 0.075,
    "min_lr_fraction": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
}

# ============================================================
# Positional encoding
# ============================================================
def get_pos_encoding(seq_length, hidden_size, device):
    return generate_positional_encoding(seq_length, hidden_size).to(device)

# ============================================================
# RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def build_norm(dim, variant_name):
    if variant_name == "pre_rms":
        return RMSNorm(dim)
    # pre_ln and post_ln both use LayerNorm
    return nn.LayerNorm(dim)

# ============================================================
# Attention
# ============================================================
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

        Q = self.WQ(H).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.WK(H).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.WV(H).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.d_head)
        return self.WO(attn_out)

# ============================================================
# Transformer block
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, d, num_heads, dropout, variant):
        super().__init__()
        self.variant = variant

        self.norm1 = build_norm(d, variant)
        self.norm2 = build_norm(d, variant)

        self.MHA = MultipleAttentionHead(d, num_heads, dropout)
        self.MLP = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )

        self.drop_attn = nn.Dropout(dropout)
        self.drop_mlp = nn.Dropout(dropout)

    def forward(self, H):
        if self.variant in ("pre_ln", "pre_rms"):
            # Pre-norm:
            # H = H + MHA(norm(H))
            # H = H + MLP(norm(H))
            H = H + self.drop_attn(self.MHA(self.norm1(H)))
            H = H + self.drop_mlp(self.MLP(self.norm2(H)))
            return H

        elif self.variant == "post_ln":
            # Post-norm:
            # H = norm(H + MHA(H))
            # H = norm(H + MLP(H))
            H = self.norm1(H + self.drop_attn(self.MHA(H)))
            H = self.norm2(H + self.drop_mlp(self.MLP(H)))
            return H

        else:
            raise ValueError(f"Unknown VARIANT: {self.variant}")

# ============================================================
# Decoder
# ============================================================
class Transformer_decoder(nn.Module):
    def __init__(self, d, num_heads, num_blocks, seq_length, dropout, variant):
        super().__init__()
        self.variant = variant
        self.TR_Blocks = nn.ModuleList(
            [TransformerBlock(d, num_heads, dropout, variant) for _ in range(num_blocks)]
        )

        # keep final norm for stability in post-norm as your note suggests
        if variant == "pre_rms":
            self.final_norm = RMSNorm(d)
        else:
            self.final_norm = nn.LayerNorm(d)

    def forward(self, batch_seq, pos_enc):
        H = batch_seq.transpose(1, 0)    # (bs, seq, d)
        H = H + pos_enc.unsqueeze(0)

        for TR_Block in self.TR_Blocks:
            H = TR_Block(H)

        H = self.final_norm(H)
        H = H.permute(1, 0, 2)           # (seq, bs, d)
        return H

# ============================================================
# Wrapper
# ============================================================
class ANN(nn.Module):
    def __init__(self, d, num_heads, num_blocks, seq_length, dropout, variant):
        super().__init__()
        self.decoder = Transformer_decoder(d, num_heads, num_blocks, seq_length, dropout, variant)

    def forward(self, g_seq, pos):
        return self.decoder(g_seq, pos)

# ============================================================
# Full LM
# ============================================================
class attention_net(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def __init__(self, vocab_size, d, num_heads, num_blocks, seq_length, dropout):
        super().__init__()
        self.layer1 = nn.Embedding(vocab_size, d)
        self.layer2 = ANN(d, num_heads, num_blocks, seq_length, dropout, VARIANT)
        self.layer3 = nn.Linear(d, vocab_size, bias=False)
        self.layer3.weight = self.layer1.weight

        self.apply(self._init_weights)

        residual_std = 0.02 / math.sqrt(2 * num_blocks)
        for block in self.layer2.decoder.TR_Blocks:
            nn.init.normal_(block.MHA.WO.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.MLP[3].weight, mean=0.0, std=residual_std)

    def forward(self, word_seq, pos):
        g_seq = self.layer1(word_seq)
        h_seq = self.layer2(g_seq, pos)
        score_seq = self.layer3(h_seq)
        return score_seq