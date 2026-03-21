import torch
import json
import math
import os
import time
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset


def generate_positional_encoding(seq_length, dim):
    assert dim == 2 * (dim // 2)  # check if dim is divisible by 2
    pe = torch.zeros(seq_length, dim)
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param / 1e6)
    )


def streaming_token_batcher(dataset_iter, tokenizer, batch_size, seq_length, eos_token_id):
    """
    Yields (data, target) tuples of shape (seq_length, batch_size).
    Concatenates tokenized documents (separated by EOS) into a flat buffer,
    then carves out chunks matching the tensor contract the model expects.
    """
    buffer = []
    chunk_size = batch_size * (seq_length + 1)  # +1 for the target offset

    for example in dataset_iter:
        tokens = tokenizer.encode(example["text"])
        tokens.append(eos_token_id)
        buffer.extend(tokens)

        while len(buffer) >= chunk_size:
            chunk = torch.LongTensor(buffer[:chunk_size])
            buffer = buffer[chunk_size:]

            chunk = chunk.view(batch_size, seq_length + 1)
            data   = chunk[:, :seq_length].t().contiguous()    # (seq_length, batch_size)
            target = chunk[:, 1:seq_length+1].t().contiguous() # (seq_length, batch_size)
            yield data, target


def get_epoch_batches(tokens_per_epoch, batch_size, seq_length, tokenizer, eos_token_id,
                      dataset_name="HuggingFaceFW/fineweb-edu",
                      subset="sample-10BT", split="train", seed=0,
                      max_retries=5):
    """Stream one epoch's worth of training batches, with retry on network errors."""
    batches_per_epoch = tokens_per_epoch // (batch_size * seq_length)
    batches_yielded = 0

    while batches_yielded < batches_per_epoch:
        try:
            ds = load_dataset(dataset_name, name=subset, split=split, streaming=True)
            ds = ds.shuffle(seed=seed)
            batcher = streaming_token_batcher(ds, tokenizer, batch_size, seq_length, eos_token_id)

            for i, (data, target) in enumerate(batcher):
                if i < batches_yielded:
                    continue
                if batches_yielded >= batches_per_epoch:
                    break
                yield data, target
                batches_yielded += 1

            break  # finished normally

        except (RuntimeError, ConnectionError, OSError) as e:
            max_retries -= 1
            if max_retries <= 0:
                raise
            print(f"  [data] stream error: {e}, retrying ({max_retries} left)...")
            time.sleep(10)
        finally:
            try:
                batcher.close()
            except:
                pass
            del ds
            gc.collect()


def load_eval_buffer(tokenizer, batch_size, eos_token_id, num_tokens=500_000,
                     dataset_name="HuggingFaceFW/fineweb-edu",
                     subset="sample-10BT", split="train"):
    """Pre-load a small eval buffer from a different region of the stream."""
    ds = load_dataset(dataset_name, name=subset, split=split, streaming=True)
    # skip ahead to avoid overlap with training data
    ds = ds.skip(50_000)
    buffer = []
    for example in ds:
        tokens = tokenizer.encode(example["text"])
        tokens.append(eos_token_id)
        buffer.extend(tokens)
        if len(buffer) >= num_tokens:
            break
    buffer = buffer[:num_tokens]
    del ds
    gc.collect()
    data = torch.LongTensor(buffer)
    # batchify: same layout as PTB
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data


def save_metrics(output_dir, config, train_ppls, eval_ppls, tokens_per_sec, peak_memory_gb):
    """Save training metrics and config to JSON for later analysis."""
    metrics = {
        "config": config,
        "train_ppls": train_ppls,
        "eval_ppls": eval_ppls,
        "tokens_per_sec": tokens_per_sec,
        "peak_memory_gb": peak_memory_gb,
        "best_eval_ppl": min(eval_ppls),
        "best_epoch": int(eval_ppls.index(min(eval_ppls))),
    }
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")


def save_plots(output_dir, train_ppls, eval_ppls, tokens_per_sec):
    """Save training curves to PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(train_ppls, label='train')
    axes[0].plot(eval_ppls, label='eval')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Perplexity')
    axes[0].legend()
    axes[0].set_title('Perplexity')

    axes[1].plot([math.log(p) for p in train_ppls], label='train')
    axes[1].plot([math.log(p) for p in eval_ppls], label='eval')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Log Loss')
    axes[1].legend()
    axes[1].set_title('Loss Curve')

    axes[2].plot(tokens_per_sec)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Tokens/sec')
    axes[2].set_title('Throughput')

    plt.tight_layout()
    path = os.path.join(output_dir, "curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plots to {path}")
