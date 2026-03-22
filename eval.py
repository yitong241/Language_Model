"""
Shared evaluation script for all ablation experiments.
Generates text from a trained checkpoint using various decoding strategies.

Usage:
    python eval.py --model-dir baseline --checkpoint baseline/runs/<jobid>/best_model.pt
    python eval.py --model-dir pe_rope --checkpoint pe_rope/runs/<jobid>/best_model.pt
    python eval.py --model-dir baseline --checkpoint <path> --prompt "Custom prompt"
    python eval.py --model-dir baseline --checkpoint <path> --max-tokens 200
"""

import argparse
import importlib.util
import math
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def load_model_module(model_dir):
    """Dynamically import model.py from the given ablation directory."""
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


@torch.no_grad()
def generate(net, tokenizer, prompt, pos_enc_fn, hidden_size, seq_length,
             eos_token_id, device, max_new_tokens=150, temperature=1.0,
             top_k=0, top_p=0.0):
    """Autoregressive text generation from a prompt string."""
    net.eval()
    token_ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        context = token_ids[-seq_length:]
        x = torch.LongTensor(context).unsqueeze(1).to(device)  # (ctx_len, 1)
        pos_enc = pos_enc_fn(x.size(0), hidden_size, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = net(x, pos_enc)       # (ctx_len, 1, vocab_size)
        logits = scores[-1, 0, :].float()  # last position logits

        # temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # top-k filtering
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[-1]] = float('-inf')

        # top-p (nucleus) filtering
        if top_p > 0.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=0), dim=0)
            remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=0) >= top_p
            sorted_logits[remove_mask] = float('-inf')
            logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=0)

        if temperature == 0 or (top_k == 0 and top_p == 0.0):
            next_token = torch.argmax(probs).item()
        else:
            next_token = torch.multinomial(probs, 1).item()

        if next_token == eos_token_id:
            break

        token_ids.append(next_token)

    return tokenizer.decode(token_ids)


def main(model_dir, checkpoint_path, custom_prompt=None, max_tokens=150):
    # ── Load model module ────────────────────────────────────────────────────
    model_module = load_model_module(model_dir)
    CONFIG = model_module.CONFIG
    ModelClass = model_module.attention_net
    get_pos_encoding = model_module.get_pos_encoding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    # ── Load model ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    eos_token_id = tokenizer.eos_token_id

    hidden_size = CONFIG["hidden_size"]
    seq_length = CONFIG["seq_length"]

    net = ModelClass(
        vocab_size, hidden_size, CONFIG["num_heads"],
        CONFIG["num_blocks"], seq_length, CONFIG["dropout"],
    )
    net = net.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    epoch = checkpoint.get('epoch', '?')
    eval_loss = checkpoint.get('eval_loss', None)
    eval_ppl = f"{math.exp(eval_loss):.2f}" if eval_loss else "?"
    print(f"Model: {CONFIG['model']}")
    print(f"Loaded checkpoint: epoch {epoch}, eval ppl {eval_ppl}")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")

    # ── Generation helper ────────────────────────────────────────────────────
    def gen(prompt, **kwargs):
        return generate(net, tokenizer, prompt, get_pos_encoding,
                        hidden_size, seq_length, eos_token_id, device,
                        max_new_tokens=max_tokens, **kwargs)

    # ── Custom prompt mode ───────────────────────────────────────────────────
    if custom_prompt:
        print(f"\n{'=' * 70}")
        print(f"PROMPT: {custom_prompt}")
        print('-' * 70)
        print("\n[Greedy]")
        print(gen(custom_prompt))
        print("\n[Top-k=50, temp=0.8]")
        print(gen(custom_prompt, temperature=0.8, top_k=50))
        print("\n[Nucleus p=0.9, temp=0.9]")
        print(gen(custom_prompt, temperature=0.9, top_p=0.9))
        return

    # ── Default prompts ──────────────────────────────────────────────────────
    prompts = [
        "The capital of France is",
        "Water freezes at a temperature of",
        "The theory of evolution was proposed by",
        "1, 2, 3, 4, 5,",
        "Monday, Tuesday, Wednesday,",
        "Photosynthesis is the process by which",
        "The three branches of the United States government are",
        "Once upon a time, in a land far away,",
        "The most important lesson I learned was",
    ]

    strategies = [
        ("Greedy",                  dict()),
        ("Top-k=50, temp=0.8",     dict(temperature=0.8, top_k=50)),
        ("Nucleus p=0.9, temp=0.9", dict(temperature=0.9, top_p=0.9)),
    ]

    for prompt in prompts:
        print(f"\n{'=' * 70}")
        print(f"PROMPT: {prompt}")
        print('-' * 70)
        for name, kwargs in strategies:
            print(f"\n[{name}]")
            print(gen(prompt, **kwargs))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained transformer model")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Ablation directory containing model.py (e.g., 'baseline', 'pe_rope')")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (if omitted, uses default prompt set)")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Maximum tokens to generate (default: 150)")
    args = parser.parse_args()
    main(args.model_dir, args.checkpoint, custom_prompt=args.prompt, max_tokens=args.max_tokens)
