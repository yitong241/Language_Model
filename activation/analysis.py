#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def import_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_model_specs(items: List[str]) -> Dict[str, Path]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path.strip()).expanduser().resolve()
    return out


def checkpoint_to_ablation_dir(ckpt_path: Path) -> Path:
    return ckpt_path.parent.parent.parent


def infer_vocab_size_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    for key in ["layer1.weight", "layer3.weight", "module.layer1.weight", "module.layer3.weight"]:
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise RuntimeError("Could not infer vocab size from checkpoint.")


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    ablation_dir = checkpoint_to_ablation_dir(ckpt_path)
    model_py = ablation_dir / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"Missing model.py: {model_py}")

    mod = import_module_from_file(f"model_{ablation_dir.name}", model_py)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v

    vocab_size = infer_vocab_size_from_state_dict(cleaned)
    cfg = mod.CONFIG

    model = mod.attention_net(
        vocab_size=vocab_size,
        d=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        num_blocks=cfg["num_blocks"],
        seq_length=cfg["seq_length"],
        dropout=cfg["dropout"],
    )

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing keys for {ckpt_path.name}: {missing[:8]}")
    if unexpected:
        print(f"[WARN] Unexpected keys for {ckpt_path.name}: {unexpected[:8]}")

    model.to(device)
    model.eval()
    return model, mod, cfg


def load_tokenizer(tokenizer_name: str):
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    tok.model_max_length = int(1e30)
    return tok


def load_eval_data(utils_py: Path, tokenizer_name: str, batch_size: int, num_tokens: int):
    utils_mod = import_module_from_file("analysis_utils_module", utils_py)

    if not hasattr(utils_mod, "load_eval_buffer"):
        raise RuntimeError(f"{utils_py} does not define load_eval_buffer(...)")

    tokenizer = load_tokenizer(tokenizer_name)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id")

    data = utils_mod.load_eval_buffer(
        tokenizer=tokenizer,
        batch_size=batch_size,
        eos_token_id=eos_token_id,
        num_tokens=num_tokens,
    )

    if not torch.is_tensor(data):
        raise RuntimeError("load_eval_buffer must return a torch tensor")
    if data.ndim != 2:
        raise RuntimeError(f"Expected eval buffer shape [T, B], got {tuple(data.shape)}")

    return data.long(), tokenizer


def iter_eval_batches(data: torch.Tensor, seq_length: int, num_batches: int):
    t_total = data.size(0)
    produced = 0
    for start in range(0, t_total - 1 - seq_length, seq_length):
        if produced >= num_batches:
            break
        x = data[start:start + seq_length]
        yield x
        produced += 1


@dataclass
class LayerStats:
    model_name: str
    layer_idx: int
    layer_name: str
    num_units: int
    dead_pct: float
    near_zero_pct: float
    mean: float
    std: float
    min_val: float
    max_val: float
    always_off_gate_pct: Optional[float] = None
    selective_gate_pct: Optional[float] = None


class StreamingLayerStats:
    def __init__(
        self,
        num_units: int,
        near_zero_threshold: float,
        gate_off_threshold: float,
        max_hist_samples: int = 200_000,
    ):
        self.num_units = num_units
        self.near_zero_threshold = near_zero_threshold
        self.gate_off_threshold = gate_off_threshold
        self.max_hist_samples = max_hist_samples

        self.ever_nonzero = torch.zeros(num_units, dtype=torch.bool)

        self.total_count = 0
        self.near_zero_count = 0

        self.sum_val = 0.0
        self.sum_sq = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")

        self.gate_ever_on: Optional[torch.Tensor] = None
        self.gate_ever_off: Optional[torch.Tensor] = None

        self.hist_sample_parts: List[np.ndarray] = []
        self.hist_sample_count = 0

    def update_hidden(self, acts: torch.Tensor):
        acts = acts.detach().float().cpu().reshape(-1, acts.shape[-1])
        threshold = 1e-4
        self.ever_nonzero |= (acts.abs() > threshold).any(dim=0)
        self.total_count += acts.numel()
        self.near_zero_count += (acts.abs() < self.near_zero_threshold).sum().item()

        self.sum_val += acts.sum().item()
        self.sum_sq += (acts * acts).sum().item()
        self.min_val = min(self.min_val, acts.min().item())
        self.max_val = max(self.max_val, acts.max().item())

        remaining = self.max_hist_samples - self.hist_sample_count
        if remaining > 0:
            flat = acts.reshape(-1)
            take = min(remaining, flat.numel())
            if take > 0:
                if take == flat.numel():
                    sample = flat.numpy()
                else:
                    idx = torch.randperm(flat.numel())[:take]
                    sample = flat[idx].numpy()
                self.hist_sample_parts.append(sample)
                self.hist_sample_count += take

    def update_gate(self, gate: torch.Tensor):
        gate = gate.detach().float().cpu().reshape(-1, gate.shape[-1])

        if self.gate_ever_on is None:
            self.gate_ever_on = torch.zeros(self.num_units, dtype=torch.bool)
            self.gate_ever_off = torch.zeros(self.num_units, dtype=torch.bool)

        self.gate_ever_on |= (gate >= self.gate_off_threshold).any(dim=0)
        self.gate_ever_off |= (gate < self.gate_off_threshold).any(dim=0)

    def finalize(self):
        dead_pct = (~self.ever_nonzero).float().mean().item() * 100.0
        near_zero_pct = 100.0 * self.near_zero_count / max(1, self.total_count)

        mean = self.sum_val / max(1, self.total_count)
        var = self.sum_sq / max(1, self.total_count) - mean * mean
        std = max(var, 0.0) ** 0.5

        always_off_gate_pct = None
        selective_gate_pct = None

        if self.gate_ever_on is not None and self.gate_ever_off is not None:
            always_off = ~self.gate_ever_on
            selective = self.gate_ever_on & self.gate_ever_off
            always_off_gate_pct = always_off.float().mean().item() * 100.0
            selective_gate_pct = selective.float().mean().item() * 100.0

        if self.hist_sample_parts:
            hist_data = np.concatenate(self.hist_sample_parts)
        else:
            hist_data = np.array([], dtype=np.float32)

        return {
            "dead_pct": dead_pct,
            "near_zero_pct": near_zero_pct,
            "mean": mean,
            "std": std,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "always_off_gate_pct": always_off_gate_pct,
            "selective_gate_pct": selective_gate_pct,
            "hist_data": hist_data,
        }


class ActivationCollector:
    def __init__(
        self,
        model_name: str,
        near_zero_threshold: float = 1e-6,
        gate_off_threshold: float = 1e-3,
        max_hist_samples_per_layer: int = 200_000,
    ):
        self.model_name = model_name
        self.near_zero_threshold = near_zero_threshold
        self.gate_off_threshold = gate_off_threshold
        self.max_hist_samples_per_layer = max_hist_samples_per_layer

        self.layer_stats: Dict[str, StreamingLayerStats] = {}
        self.handles = []

    def _ensure_layer(self, layer_name: str, num_units: int):
        if layer_name not in self.layer_stats:
            self.layer_stats[layer_name] = StreamingLayerStats(
                num_units=num_units,
                near_zero_threshold=self.near_zero_threshold,
                gate_off_threshold=self.gate_off_threshold,
                max_hist_samples=self.max_hist_samples_per_layer,
            )

    def attach(self, model: nn.Module):
        blocks = model.layer2.decoder.TR_Blocks

        for i, block in enumerate(blocks):
            layer_name = f"TR_Blocks.{i}.MLP"
            mlp = block.MLP

            if hasattr(mlp, "w1") and hasattr(mlp, "w2") and hasattr(mlp, "w3"):
                def make_swiglu_hook(name):
                    def hook(module, inputs, output):
                        x = inputs[0]
                        value = module.w1(x)
                        gate = F.silu(module.w2(x))
                        hidden = value * gate

                        self._ensure_layer(name, hidden.shape[-1])
                        self.layer_stats[name].update_hidden(hidden)
                        self.layer_stats[name].update_gate(gate)
                    return hook

                self.handles.append(mlp.register_forward_hook(make_swiglu_hook(layer_name)))
                continue

            if isinstance(mlp, nn.Sequential):
                act_idx = None
                for j, sub in enumerate(mlp):
                    if isinstance(sub, (nn.ReLU, nn.GELU, nn.SiLU)):
                        act_idx = j
                        break

                if act_idx is None:
                    raise RuntimeError(f"Could not find activation in {layer_name}")

                act_module = mlp[act_idx]

                def make_act_hook(name):
                    def hook(module, inputs, output):
                        self._ensure_layer(name, output.shape[-1])
                        self.layer_stats[name].update_hidden(output)
                    return hook

                self.handles.append(act_module.register_forward_hook(make_act_hook(layer_name)))
                continue

            raise RuntimeError(f"Unsupported MLP structure in {layer_name}: {type(mlp)}")

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def compute(self) -> Tuple[List[LayerStats], Dict[str, np.ndarray]]:
        stats = []
        hist_data = {}

        for layer_name in sorted(self.layer_stats.keys(), key=lambda s: int(s.split(".")[1])):
            result = self.layer_stats[layer_name].finalize()
            layer_idx = int(layer_name.split(".")[1])
            num_units = self.layer_stats[layer_name].num_units

            stats.append(
                LayerStats(
                    model_name=self.model_name,
                    layer_idx=layer_idx,
                    layer_name=layer_name,
                    num_units=num_units,
                    dead_pct=result["dead_pct"],
                    near_zero_pct=result["near_zero_pct"],
                    mean=result["mean"],
                    std=result["std"],
                    min_val=result["min_val"],
                    max_val=result["max_val"],
                    always_off_gate_pct=result["always_off_gate_pct"],
                    selective_gate_pct=result["selective_gate_pct"],
                )
            )

            hist_data[layer_name] = result["hist_data"]

        return stats, hist_data


def analyze_one_model(
    model_name: str,
    ckpt_path: Path,
    device: torch.device,
    eval_data: torch.Tensor,
    num_batches: int,
    near_zero_threshold: float,
    gate_off_threshold: float,
    max_hist_samples_per_layer: int,
):
    model, mod, cfg = load_model_from_checkpoint(ckpt_path, device)

    collector = ActivationCollector(
        model_name=model_name,
        near_zero_threshold=near_zero_threshold,
        gate_off_threshold=gate_off_threshold,
        max_hist_samples_per_layer=max_hist_samples_per_layer,
    )
    collector.attach(model)

    seq_length = cfg["seq_length"]

    with torch.no_grad():
        pos = mod.get_pos_encoding(seq_length, cfg["hidden_size"], device)
        for x in iter_eval_batches(eval_data, seq_length, num_batches):
            x = x.to(device, non_blocking=True)
            _ = model(x, pos)

    collector.remove()
    stats, hist_data = collector.compute()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats, hist_data


def save_csv(all_stats: Dict[str, List[LayerStats]], output_dir: Path):
    with open(output_dir / "layerwise_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "layer_idx", "layer_name", "num_units",
            "dead_pct", "near_zero_pct", "mean", "std", "min_val", "max_val",
            "always_off_gate_pct", "selective_gate_pct",
        ])
        for model_name in sorted(all_stats.keys()):
            for s in all_stats[model_name]:
                writer.writerow([
                    s.model_name, s.layer_idx, s.layer_name, s.num_units,
                    f"{s.dead_pct:.6f}", f"{s.near_zero_pct:.6f}",
                    f"{s.mean:.6f}", f"{s.std:.6f}",
                    f"{s.min_val:.6f}", f"{s.max_val:.6f}",
                    "" if s.always_off_gate_pct is None else f"{s.always_off_gate_pct:.6f}",
                    "" if s.selective_gate_pct is None else f"{s.selective_gate_pct:.6f}",
                ])


def save_json(all_stats: Dict[str, List[LayerStats]], output_dir: Path):
    summary = {}
    for model_name, stats in all_stats.items():
        summary[model_name] = {
            "num_layers": len(stats),
            "avg_dead_pct": float(np.mean([s.dead_pct for s in stats])) if stats else None,
            "avg_near_zero_pct": float(np.mean([s.near_zero_pct for s in stats])) if stats else None,
            "avg_always_off_gate_pct": (
                float(np.mean([s.always_off_gate_pct for s in stats if s.always_off_gate_pct is not None]))
                if any(s.always_off_gate_pct is not None for s in stats) else None
            ),
            "avg_selective_gate_pct": (
                float(np.mean([s.selective_gate_pct for s in stats if s.selective_gate_pct is not None]))
                if any(s.selective_gate_pct is not None for s in stats) else None
            ),
            "layers": [asdict(s) for s in stats],
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def plot_metric(all_stats: Dict[str, List[LayerStats]], metric: str, ylabel: str, title: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    for model_name, stats in sorted(all_stats.items()):
        xs = [s.layer_idx for s in stats]
        ys = [getattr(s, metric) if getattr(s, metric) is not None else np.nan for s in stats]
        plt.plot(xs, ys, marker="o", label=model_name)
    plt.xlabel("Layer depth")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_histograms(all_hist: Dict[str, Dict[str, np.ndarray]], output_dir: Path, bins: int = 200):
    hist_dir = output_dir / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    for model_name, layer_map in all_hist.items():
        non_empty = [v for v in layer_map.values() if v.size > 0]
        if not non_empty:
            continue

        combined = np.concatenate(non_empty)
        plt.figure(figsize=(8, 5))
        plt.hist(combined, bins=bins, log=True)
        plt.xlabel("Activation value")
        plt.ylabel("Count (log scale)")
        plt.title(f"{model_name}: all FFN hidden activations")
        plt.tight_layout()
        plt.savefig(hist_dir / f"{model_name}_all_layers.png", dpi=220)
        plt.close()

        layer_names = sorted(layer_map.keys(), key=lambda s: int(s.split(".")[1]))
        picks = []
        if len(layer_names) >= 1:
            picks.append(layer_names[0])
        if len(layer_names) >= 3:
            picks.append(layer_names[len(layer_names) // 2])
        if len(layer_names) >= 2:
            picks.append(layer_names[-1])

        seen = set()
        for lname in picks:
            if lname in seen or layer_map[lname].size == 0:
                continue
            seen.add(lname)

            plt.figure(figsize=(8, 5))
            plt.hist(layer_map[lname], bins=bins, log=True)
            plt.xlabel("Activation value")
            plt.ylabel("Count (log scale)")
            plt.title(f"{model_name}: {lname}")
            plt.tight_layout()
            plt.savefig(hist_dir / f"{model_name}_{lname.replace('.', '_')}.png", dpi=220)
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Examples: relu=/path/to/baseline.pt gelu=/path/to/act_gelu.pt swiglu=/path/to/act_swiglu.pt",
    )
    parser.add_argument("--utils-py", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-num-tokens", type=int, default=150000)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--near-zero-threshold", type=float, default=1e-6)
    parser.add_argument("--gate-off-threshold", type=float, default=1e-3)
    parser.add_argument("--max-hist-samples-per-layer", type=int, default=200000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = parse_model_specs(args.models)

    eval_data, _tokenizer = load_eval_data(
        utils_py=Path(args.utils_py).expanduser().resolve(),
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size,
        num_tokens=args.eval_num_tokens,
    )

    print(f"Loaded eval buffer shape: {tuple(eval_data.shape)}")
    print(f"Using device: {device}")

    all_stats: Dict[str, List[LayerStats]] = {}
    all_hist: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name, ckpt_path in model_specs.items():
        print(f"\n=== analyzing {model_name} ===")
        print(f"checkpoint: {ckpt_path}")

        stats, hist = analyze_one_model(
            model_name=model_name,
            ckpt_path=ckpt_path,
            device=device,
            eval_data=eval_data,
            num_batches=args.num_batches,
            near_zero_threshold=args.near_zero_threshold,
            gate_off_threshold=args.gate_off_threshold,
            max_hist_samples_per_layer=args.max_hist_samples_per_layer,
        )

        all_stats[model_name] = stats
        all_hist[model_name] = hist

        for s in stats:
            extra = ""
            if s.always_off_gate_pct is not None:
                extra = f", always_off_gate={s.always_off_gate_pct:.3f}%, selective_gate={s.selective_gate_pct:.3f}%"
            print(
                f"layer {s.layer_idx:02d}: dead={s.dead_pct:.3f}%, near_zero={s.near_zero_pct:.3f}%, "
                f"mean={s.mean:.4e}, std={s.std:.4e}{extra}"
            )

    save_csv(all_stats, output_dir)
    save_json(all_stats, output_dir)

    plot_metric(
        all_stats,
        metric="dead_pct",
        ylabel="Dead neurons (%)",
        title="Dead neuron % vs layer depth",
        out_path=output_dir / "dead_neurons_vs_depth.png",
    )
    plot_metric(
        all_stats,
        metric="near_zero_pct",
        ylabel="Near-zero activations (%)",
        title="Near-zero activation % vs layer depth",
        out_path=output_dir / "near_zero_vs_depth.png",
    )

    if any(any(s.always_off_gate_pct is not None for s in stats) for stats in all_stats.values()):
        plot_metric(
            all_stats,
            metric="always_off_gate_pct",
            ylabel="Always-off gate (%)",
            title="SwiGLU always-off gate % vs layer depth",
            out_path=output_dir / "swiglu_always_off_gate_vs_depth.png",
        )
        plot_metric(
            all_stats,
            metric="selective_gate_pct",
            ylabel="Selective gate (%)",
            title="SwiGLU selective gate % vs layer depth",
            out_path=output_dir / "swiglu_selective_gate_vs_depth.png",
        )

    plot_histograms(all_hist, output_dir)

    print(f"\nSaved results to: {output_dir}")
    print("  layerwise_stats.csv")
    print("  summary.json")
    print("  dead_neurons_vs_depth.png")
    print("  near_zero_vs_depth.png")
    print("  histograms/*.png")


if __name__ == "__main__":
    main()