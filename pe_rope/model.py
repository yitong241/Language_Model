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

# ── Config ──────────────────────────────────────────────────────────────────────

CONFIG = {
    "model": "pe_rope",
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
    """
    RoPE rotates Q and K inside attention rather than adding a fixed offset
    to token embeddings. Typical return: (cos, sin) each of shape
    (seq_length, d_head), precomputed on `device`.
    """
    assert hidden_size == 2 * (hidden_size // 2)  # check if dim is divisible by 2
    d_head = hidden_size // CONFIG['num_heads']
    return precompute_rope_freqs(seq_length, d_head, device)

def precompute_rope_freqs(seq_length, d_head, device, base=10000):
    # Calculate rotation angle for each index
    # aka computes cos(mθ) and sin(mθ)
    theta = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device) / d_head))
    theta = theta.repeat(2)
    pos = torch.arange(0, seq_length, device=device).unsqueeze(1)
    m_theta = pos * theta 

    return m_theta.cos(), m_theta.sin()

def rotate_half(x):
    # Turns [x1, x2, x3, x4]
    # into [-x3, -x4, x1, x2]
    # aka computes q[idx]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_embeddings(q_or_k, cos_freqs, sin_freqs):
    # Uses the modern RoPE pairing: (q_0, q_d/2), (q_1, q_d/2 + 1)...
    # q_rot[0] =  q[0]·cos(mθ₀) - q[2]·sin(mθ₀)
    # q_rot[1] =  q[1]·cos(mθ₁) + q[3]·sin(mθ₁)
    # q_rot[2] =  q[2]·cos(mθ₀) - q[0]·sin(mθ₀)
    # q_rot[3] =  q[3]·cos(mθ₁) + q[1]·sin(mθ₁)
    # Does this thing above ^, where m is the sequence index, q is the query or key
    # Note: Its different from the RoPE paper which is (q_0, q_1), (q2, q3) etc, but its just a performance
    # thing, the result is still same
    return q_or_k * cos_freqs + rotate_half(q_or_k) * sin_freqs

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

        ############ For analysis only
        self.capture_attn = False
        self.attn_weights = None
        #############

    def forward(self, H, pos_enc):
        # (RoPE): add `pos` argument (the (cos, sin) tuple from get_pos_encoding)
        batch, seq_len, _ = H.shape

        pos_enc_cos, pos_enc_sin = pos_enc

        Q = self.WQ(H).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        Q_rope = apply_rotary_embeddings(Q, pos_enc_cos, pos_enc_sin)

        K = self.WK(H).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K_rope = apply_rotary_embeddings(K, pos_enc_cos, pos_enc_sin)

        V = self.WV(H).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            Q_rope, K_rope, V, is_causal=True,
            
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.d_head * self.num_heads)

        ###### Analysis Code #####
        if self.capture_attn:
            attn_score = Q_rope @ K_rope.transpose(-2, -1) * self.d_head**-0.5 # [Batch size, Num Heads, seq_len, seq_len]
            seq_len = Q_rope.shape[2]
            mask = torch.tril(torch.ones(seq_len, seq_len)).long().to(attn_score.device)
            attn_score = attn_score.masked_fill(mask==0, value=float('-inf'))
            attn_score = torch.softmax(attn_score, dim=-1)
            self.attn_weights = attn_score
        # attn_out = attn_score @ V
        #####

        return self.WO(attn_out)

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
        self.drop_mlp  = nn.Dropout(dropout)

    def forward(self, H, pos_enc):
        # add `pos` argument and forward it: self.MHA(self.LN_MHA(H), pos)
        H = H + self.drop_attn(self.MHA(self.LN_MHA(H), pos_enc))
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
        # Pass pos_enc into each block instead: H = TR_Block(H, pos_enc)
        for TR_Block in self.TR_Blocks:
            H = TR_Block(H, pos_enc)
        H = self.final_norm(H)
        return H.permute(1, 0, 2)

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
        g_seq     = self.layer1(word_seq)
        h_seq     = self.layer2(g_seq, pos)
        score_seq = self.layer3(h_seq)
        return score_seq
