import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    "model": "pe_alibi",
    "bs": 64,
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


def get_pos_encoding(seq_length, hidden_size, device):
    """
    ALiBi uses no positional embedding tensor at all.
    Keep this function for interface compatibility.
    """
    return torch.zeros(seq_length, hidden_size, device=device)


def _get_slopes_power_of_2(n: int):
    start = 2 ** (-(2 ** -(math.log2(n) - 3)))
    ratio = start
    return [start * (ratio ** i) for i in range(n)]


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    Official ALiBi slope construction used in the paper/reference code.
    """
    if math.log2(n_heads).is_integer():
        slopes = _get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = _get_slopes_power_of_2(closest_power_of_2)
        extra = get_alibi_slopes(2 * closest_power_of_2)[0::2][: n_heads - closest_power_of_2]
        slopes = slopes + extra.tolist()
    return torch.tensor(slopes, dtype=torch.float32)


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

        slopes = get_alibi_slopes(num_heads)  # [H]
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def _build_alibi_bias(self, seq_len: int, device, dtype):
        """
        Returns additive attention bias of shape [1, H, L, L].

        - future positions get -inf (causal mask)
        - allowed past/current positions get ALiBi penalty:
              bias[h, i, j] = -slope[h] * (i - j), for j <= i
        """
        pos = torch.arange(seq_len, device=device)
        rel_dist = pos[:, None] - pos[None, :]          # [L, L], positive on/under diagonal
        rel_dist = rel_dist.clamp(min=0)

        slopes = self.alibi_slopes.to(device=device, dtype=torch.float32).view(1, self.num_heads, 1, 1)
        bias = -slopes * rel_dist.view(1, 1, seq_len, seq_len).to(torch.float32)

        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        bias = bias.masked_fill(causal.view(1, 1, seq_len, seq_len), float("-inf"))
        return bias.to(dtype)

    def forward(self, H):
        batch, seq_len, _ = H.shape

        Q = self.WQ(H).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.WK(H).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.WV(H).reshape(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        attn_bias = self._build_alibi_bias(seq_len, H.device, Q.dtype)

        attn_out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_bias,   # additive bias to logits
            is_causal=False,       # causal mask already folded into attn_bias
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.d_head * self.num_heads)
        H = self.WO(attn_out)
        return H


class TransformerBlock(nn.Module):
    def __init__(self, d, num_heads, dropout):
        super().__init__()
        self.LN_MHA = nn.LayerNorm(d)
        self.LN_MLP = nn.LayerNorm(d)
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

    def forward(self, batch_seq, pos_enc=None):
        # ALiBi: do NOT add positional embeddings
        H = batch_seq.transpose(1, 0)
        for TR_Block in self.TR_Blocks:
            H = TR_Block(H)
        H = self.final_norm(H)
        H = H.permute(1, 0, 2)
        return H


class ANN(nn.Module):
    def __init__(self, d, num_heads, num_blocks, seq_length, dropout):
        super().__init__()
        self.decoder = Transformer_decoder(d, num_heads, num_blocks, seq_length, dropout)

    def forward(self, g_seq, pos=None):
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

    def forward(self, word_seq, pos=None):
        g_seq = self.layer1(word_seq)
        h_seq = self.layer2(g_seq, pos)
        score_seq = self.layer3(h_seq)
        return score_seq