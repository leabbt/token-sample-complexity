#!/usr/bin/env python3
"""
Per-sample downstream experiment: stores ALL errors per (example, n, layer)
without averaging across examples.

Measures at full layer output (post-FFN+residual+LN), non-padded.
Also stores per-MC-rep predictions for classification analysis.

Stored arrays:
  Per (example, n, layer):  err_mean, err_cov, err_l2
  Per (example, n):         err_pool_l2, err_logit_l2, err_kl
  Per (example, n, k_mc):   sparse_preds
  Per (example):            dense_logits, labels, actual_lens

Usage:
    python run_persample.py --model results/checkpoint --results-dir results_persample
    python run_persample.py --start-idx 0 --end-idx 600
"""

import os

# Downstream defaults (formerly in config.py). Override at the CLI.
DATASET_NAME = "ccdv/arxiv-classification"
DATASET_LOCAL_DIR = None
MAX_LENGTH = 4096
MIN_TOKEN_RATIO = 0.9
N_MIN = 64
N_MAX = MAX_LENGTH
NB_POINTS = 20
K_MC = 20
BASE_SEED = 42
CHECKPOINT_DIR = None     # require --model
RESULTS_DIR = "results"
import json
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer
from model import BigBirdMeanPoolClassifier
from datasets import load_dataset


N_EXAMPLES_DEFAULT = 2400


# ── Sparse mask generation ────────────────────────────────────────────────────

def generate_sparse_mask(seq_len, n, actual_len, seed, device):
    rng = torch.Generator().manual_seed(seed)
    n_sample = min(n, actual_len)
    perm = torch.randperm(actual_len, generator=rng)[:n_sample]
    mask = torch.full((1, 1, 1, seq_len), -10000.0, device=device)
    mask[0, 0, 0, perm.to(device)] = 0.0
    return mask


# ── Hook machinery ────────────────────────────────────────────────────────────

def _make_sparse_hook(n, seq_len, actual_len, seed, device):
    def hook(module, args, kwargs):
        mask = generate_sparse_mask(seq_len, n, actual_len, seed, device)
        if len(args) > 1 and args[1] is not None:
            new_args = (args[0], args[1] + mask) + args[2:]
            return new_args, kwargs
        elif "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            kw = dict(kwargs)
            kw["attention_mask"] = kw["attention_mask"] + mask
            return args, kw
        return args, kwargs
    return hook


def _get_backbone(model):
    for attr in ("bert", "bigbird"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError("Cannot find BigBird backbone")


def install_sparse_hooks(model, n, seq_len, actual_len, base_seed, num_layers, device):
    backbone = _get_backbone(model)
    handles = []
    for l in range(num_layers):
        seed_l = base_seed + l * 7919
        attn = backbone.encoder.layer[l].attention.self
        h = attn.register_forward_pre_hook(
            _make_sparse_hook(n, seq_len, actual_len, seed_l, device),
            with_kwargs=True,
        )
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── Error helpers (non-padded) ───────────────────────────────────────────────

def mean_error(h_sparse, h_dense, actual_len):
    h_s = h_sparse[:, :actual_len, :]
    h_d = h_dense[:, :actual_len, :]
    return (h_s.mean(dim=1) - h_d.mean(dim=1)).norm().item()


def cov_error(h_sparse, h_dense, actual_len):
    def _cov(h):
        h = h.squeeze(0)[:actual_len]
        h = h - h.mean(0, keepdim=True)
        return h.T @ h / max(h.shape[0] - 1, 1)
    return (_cov(h_sparse) - _cov(h_dense)).norm().item()


def l2_error(h_sparse, h_dense, actual_len):
    per_token_sq = (h_sparse[:, :actual_len, :] - h_dense[:, :actual_len, :]).squeeze(0).pow(2).sum(dim=1)
    return per_token_sq.mean().item()


def pool_l2_error(h_sparse, h_dense, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    pool_s = (h_sparse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    pool_d = (h_dense * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return (pool_s - pool_d).norm().item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHECKPOINT_DIR)
    ap.add_argument("--dataset", default=DATASET_NAME)
    ap.add_argument("--results-dir", default="results_persample")
    ap.add_argument("--n-examples", type=int, default=N_EXAMPLES_DEFAULT)
    ap.add_argument("--k-mc", type=int, default=K_MC)
    ap.add_argument("--n-min", type=int, default=N_MIN)
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--nb-points", type=int, default=NB_POINTS)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.results_dir, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────────
    print(f"[Step 1] Loading model: {args.model}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BigBirdMeanPoolClassifier.from_pretrained(args.model)
    model.to(device).eval()

    num_labels = model.config.num_labels
    num_layers = model.config.num_hidden_layers

    backbone = _get_backbone(model)
    for layer in backbone.encoder.layer:
        layer.attention.self.attention_type = "original_full"
    model.config.attention_type = "original_full"

    print(f"  num_labels={num_labels}  num_layers={num_layers}  "
          f"loaded in {time.time()-t0:.1f}s")

    # ── 2. Load + tokenize test set ──────────────────────────────────────────
    print(f"[Step 2] Loading dataset: {args.dataset}")
    t0 = time.time()

    if DATASET_LOCAL_DIR:
        ds = load_dataset("parquet", data_dir=DATASET_LOCAL_DIR, split="test")
    else:
        ds = load_dataset(args.dataset, split="test")

    def tokenize_fn(example):
        tok = tokenizer(example["text"], max_length=MAX_LENGTH,
                        truncation=True, padding="max_length")
        return {"input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "label": example["label"]}

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    ds.set_format("torch")

    lengths = np.array([int(ex["attention_mask"].sum()) for ex in ds])
    min_len = int(MAX_LENGTH * MIN_TOKEN_RATIO)
    long_idx = np.where(lengths >= min_len)[0]

    if len(long_idx) >= args.n_examples:
        sel = long_idx[:args.n_examples]
    else:
        sel = np.arange(min(args.n_examples, len(ds)))
    ds = ds.select(sel.tolist())

    # Distributed slicing
    end_idx = args.end_idx if args.end_idx is not None else len(ds)
    end_idx = min(end_idx, len(ds))
    if args.start_idx > 0 or args.end_idx is not None:
        ds = ds.select(list(range(args.start_idx, end_idx)))
        print(f"  Slice: [{args.start_idx}, {end_idx}) = {len(ds)} examples")

    n_ex = len(ds)
    print(f"  {n_ex} examples — {time.time()-t0:.1f}s")

    # ── n grid ───────────────────────────────────────────────────────────────
    n_values = np.unique(np.append(
        np.geomspace(args.n_min, args.n_max, args.nb_points).astype(int),
        args.n_max,
    ))
    nb_n = len(n_values)
    print(f"  n grid ({nb_n} points): {n_values.tolist()}")

    shard = (f"_shard_{args.start_idx}_{end_idx}"
             if (args.start_idx > 0 or args.end_idx is not None) else "")

    # ── Pre-allocate ─────────────────────────────────────────────────────────
    dense_logits   = np.zeros((n_ex, num_labels))
    labels_arr     = np.zeros(n_ex, dtype=int)
    actual_lens    = np.zeros(n_ex, dtype=int)

    # Per-layer errors: (n_ex, nb_n, num_layers) — MC-averaged
    err_mean  = np.zeros((n_ex, nb_n, num_layers))
    err_cov   = np.zeros((n_ex, nb_n, num_layers))
    err_l2    = np.zeros((n_ex, nb_n, num_layers))

    # End-to-end errors: (n_ex, nb_n) — MC-averaged
    err_pool  = np.zeros((n_ex, nb_n))
    err_logit = np.zeros((n_ex, nb_n))
    err_kl    = np.zeros((n_ex, nb_n))

    # Per-MC predictions: (n_ex, nb_n, k_mc)
    sparse_preds = np.zeros((n_ex, nb_n, args.k_mc), dtype=np.int16)

    # ── Resume ───────────────────────────────────────────────────────────────
    raw_dir = os.path.join(args.results_dir, f"raw{shard}")
    os.makedirs(raw_dir, exist_ok=True)
    progress_file = os.path.join(raw_dir, "progress.json")

    start_example = 0
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            prog = json.load(f)
        start_example = prog.get("done", 0)
        if start_example > 0:
            dense_logits = np.load(os.path.join(raw_dir, "dense_logits.npy"))
            labels_arr   = np.load(os.path.join(raw_dir, "labels.npy"))
            actual_lens  = np.load(os.path.join(raw_dir, "actual_lens.npy"))
            err_mean     = np.load(os.path.join(raw_dir, "err_mean.npy"))
            err_cov      = np.load(os.path.join(raw_dir, "err_cov.npy"))
            err_l2       = np.load(os.path.join(raw_dir, "err_l2.npy"))
            err_pool     = np.load(os.path.join(raw_dir, "err_pool.npy"))
            err_logit    = np.load(os.path.join(raw_dir, "err_logit.npy"))
            err_kl       = np.load(os.path.join(raw_dir, "err_kl.npy"))
            sparse_preds = np.load(os.path.join(raw_dir, "sparse_preds.npy"))
            print(f"  Resuming from example {start_example}/{n_ex}")

    SAVE_EVERY = 10

    # ── 3. Inference ─────────────────────────────────────────────────────────
    print(f"\n[Step 3] Inference: {n_ex} ex × {nb_n} n-values × {args.k_mc} MC")
    t_infer = time.time()

    for i in tqdm(range(start_example, n_ex), desc="examples",
                  initial=start_example, total=n_ex):
        ids  = ds[i]["input_ids"].unsqueeze(0).to(device)
        amsk = ds[i]["attention_mask"].unsqueeze(0).to(device)
        labels_arr[i] = ds[i]["label"].item()
        seq_len    = ids.shape[1]
        actual_len = int(amsk.sum().item())
        actual_lens[i] = actual_len

        # ── Dense forward ──
        with torch.no_grad():
            d_out = model(ids, attention_mask=amsk, output_hidden_states=True)
        dense_logits[i] = d_out.logits[0].cpu().numpy()
        d_hidden = [h.clone() for h in d_out.hidden_states[1:]]  # layers 0..11 output
        d_logits_t = d_out.logits[0]  # (num_labels,) on device

        # ── Sparse forwards ──
        for j, nv in enumerate(n_values):
            if nv >= actual_len:
                # Dense = sparse for this n
                err_mean[i, j, :]  = 0.0
                err_cov[i, j, :]   = 0.0
                err_l2[i, j, :]    = 0.0
                err_pool[i, j]     = 0.0
                err_logit[i, j]    = 0.0
                err_kl[i, j]       = 0.0
                sparse_preds[i, j, :] = dense_logits[i].argmax()
                continue

            acc_layer  = np.zeros((3, num_layers))  # mean, cov, l2
            acc_pool   = 0.0
            acc_logit  = 0.0
            acc_kl     = 0.0

            for mc in range(args.k_mc):
                seed = BASE_SEED + i * 100_000 + j * 1_000 + mc
                hooks = install_sparse_hooks(
                    model, nv, seq_len, actual_len, seed,
                    num_layers, device,
                )
                with torch.no_grad():
                    s_out = model(ids, attention_mask=amsk,
                                  output_hidden_states=True)
                remove_hooks(hooks)

                s_logits = s_out.logits[0]  # (num_labels,)
                sparse_preds[i, j, mc] = int(s_logits.argmax().item())

                # Per-layer errors
                for l in range(num_layers):
                    hl_s = s_out.hidden_states[l + 1]
                    hl_d = d_hidden[l]
                    acc_layer[0, l] += mean_error(hl_s, hl_d, actual_len)
                    acc_layer[1, l] += cov_error(hl_s, hl_d, actual_len)
                    acc_layer[2, l] += l2_error(hl_s, hl_d, actual_len)

                # End-to-end
                acc_pool += pool_l2_error(
                    s_out.hidden_states[-1], d_hidden[-1], amsk)
                acc_logit += (s_logits - d_logits_t).norm().item()

                # KL(dense || sparse)
                p = torch.softmax(d_logits_t.double(), 0)
                q = torch.softmax(s_logits.double(), 0)
                acc_kl += (p * (p.clamp(min=1e-12).log() - q.clamp(min=1e-12).log())).sum().item()

            K = args.k_mc
            err_mean[i, j]  = acc_layer[0] / K
            err_cov[i, j]   = acc_layer[1] / K
            err_l2[i, j]    = acc_layer[2] / K
            err_pool[i, j]  = acc_pool / K
            err_logit[i, j] = acc_logit / K
            err_kl[i, j]    = acc_kl / K

        del d_hidden
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── Incremental save ──
        if (i + 1) % SAVE_EVERY == 0 or i == n_ex - 1:
            for name, arr in [
                ("dense_logits", dense_logits), ("labels", labels_arr),
                ("actual_lens", actual_lens),
                ("err_mean", err_mean), ("err_cov", err_cov), ("err_l2", err_l2),
                ("err_pool", err_pool), ("err_logit", err_logit), ("err_kl", err_kl),
                ("sparse_preds", sparse_preds),
            ]:
                np.save(os.path.join(raw_dir, f"{name}.npy"), arr)
            with open(progress_file, "w") as f:
                json.dump({"done": i + 1, "total": n_ex}, f)

    elapsed = time.time() - t_infer
    print(f"\nInference done in {elapsed:.0f}s ({elapsed/n_ex:.1f}s/ex)")

    # ── 4. Save final arrays ─────────────────────────────────────────────────
    print(f"[Step 4] Saving to {args.results_dir}")
    rd = args.results_dir

    np.save(os.path.join(rd, f"n_values.npy"), n_values)
    np.save(os.path.join(rd, f"dense_logits{shard}.npy"), dense_logits)
    np.save(os.path.join(rd, f"labels{shard}.npy"), labels_arr)
    np.save(os.path.join(rd, f"actual_lens{shard}.npy"), actual_lens)
    np.save(os.path.join(rd, f"err_mean{shard}.npy"), err_mean)
    np.save(os.path.join(rd, f"err_cov{shard}.npy"), err_cov)
    np.save(os.path.join(rd, f"err_l2{shard}.npy"), err_l2)
    np.save(os.path.join(rd, f"err_pool{shard}.npy"), err_pool)
    np.save(os.path.join(rd, f"err_logit{shard}.npy"), err_logit)
    np.save(os.path.join(rd, f"err_kl{shard}.npy"), err_kl)
    np.save(os.path.join(rd, f"sparse_preds{shard}.npy"), sparse_preds)

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "n_values": n_values.tolist(),
        "n_examples": n_ex,
        "k_mc": args.k_mc,
        "num_layers": num_layers,
        "num_labels": num_labels,
        "max_length": MAX_LENGTH,
        "start_idx": args.start_idx,
        "end_idx": end_idx,
        "elapsed_s": elapsed,
        "dense_accuracy": float((dense_logits.argmax(1) == labels_arr).mean()),
    }
    with open(os.path.join(rd, f"meta{shard}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Dense accuracy: {meta['dense_accuracy']:.4f}")


if __name__ == "__main__":
    main()
