#!/usr/bin/env python3
"""
Compute horizon H and covariance statistics per (example, layer) for the
downstream BigBird model.

For each test example and each layer l, with Σ_l = LL^T (Cholesky, L lower triangular):
  - X_l = hidden states before attention (input to layer l)
  - Σ_l = token covariance of X_l (non-padded tokens only)
  - λ_max(Σ_l), tr(Σ_l)
  - H_l     = ||L^T (Wk^T Wq) L||_2  ( = ||Σ_l^{1/2} A Σ_l^{1/2}||_2 )
                  Mirrors experiments/horizons/compute_horizons.py:Horizon_new.
  - H_l^{(h)} per attention head, same two-sided formula on A_h = Wk_h^T Wq_h.

Usage:
    python compute_horizon.py --model results/checkpoint --results-dir results_horizon
    # Distributed:
    python compute_horizon.py --start-idx 0 --end-idx 600
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



def _get_backbone(model):
    for attr in ("bert", "bigbird"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError("Cannot find BigBird backbone")


def extract_qkv_weights(model, layer_id):
    """Extract Wq, Wk weight matrices for a given layer."""
    backbone = _get_backbone(model)
    attn = backbone.encoder.layer[layer_id].attention.self
    Wq = attn.query.weight.detach()  # (768, 768)
    Wk = attn.key.weight.detach()    # (768, 768)
    return Wq, Wk


def compute_sigma_stats(X, actual_len):
    """
    Compute covariance statistics from hidden states.

    Parameters
    ----------
    X : (1, seq_len, d) tensor — hidden states for one example
    actual_len : int — number of real (non-padded) tokens

    Returns
    -------
    cov : (d, d) tensor — token covariance
    lam_max : float — max eigenvalue of cov
    trace : float — trace of cov
    mean : (d,) tensor — token mean
    """
    x = X[0, :actual_len, :]  # (actual_len, d)
    n = x.shape[0]
    mean = x.mean(dim=0)  # (d,)
    m2 = (x.T @ x) / n   # (d, d)
    cov = m2 - mean.unsqueeze(1) @ mean.unsqueeze(0)  # (d, d)

    eigenvalues = torch.linalg.eigvalsh(cov)  # sorted ascending
    lam_max = eigenvalues[-1].item()
    trace = cov.trace().item()

    return cov, lam_max, trace, mean


def compute_horizon(cov, Wq, Wk, num_heads=12):
    """Two-sided horizon ``H = ‖Lᵀ A L‖₂`` via Cholesky of ``Σ + 1e-6·I``.

    ``L`` is the lower-triangular Cholesky factor of ``cov + 1e-6·I``
    (so ``L Lᵀ = cov + 1e-6·I``). The aggregated horizon takes ``A = WkᵀWq``
    and returns ``‖Lᵀ A L‖₂`` directly. Per-head, the explicit ``(d, d)``
    matrix ``A_h = Wk_hᵀ Wq_h`` has rank ≤ head_dim, and combined with the
    cyclic-singular-value identity ``σ(BC) = σ(CB)`` we reduce the per-head
    eigenproblem to a ``(head_dim, head_dim)`` one:

        ‖Lᵀ A_h L‖₂² = λ_max((Lᵀ A_h L)(Lᵀ A_h L)ᵀ)
                     = λ_max(Lᵀ A_h Σ A_hᵀ L)        (L Lᵀ = Σ)
                     = λ_max(A_h Σ A_hᵀ Σ)           (cyclic)
                     = λ_max((Wq_h Σ Wq_hᵀ)(Wk_h Σ Wk_hᵀ))
    """
    d = cov.shape[0]
    head_dim = d // num_heads

    cov64 = cov.to(torch.float64)
    cov64 = 0.5 * (cov64 + cov64.transpose(-1, -2))
    cov_reg = cov64 + 1e-6 * torch.eye(d, device=cov64.device, dtype=cov64.dtype)
    L = torch.linalg.cholesky(cov_reg)                            # lower-triangular

    Wq64 = Wq.to(torch.float64)
    Wk64 = Wk.to(torch.float64)

    # Aggregated horizon
    A_agg = Wk64.T @ Wq64
    H_agg = torch.linalg.matrix_norm(L.T @ A_agg @ L, ord=2).item()

    # Per-head horizons via the cyclic identity. cov_full ≡ L Lᵀ = Σ + εI.
    cov_full = L @ L.T
    Wq_h_all = Wq64.view(num_heads, head_dim, d)
    Wk_h_all = Wk64.view(num_heads, head_dim, d)
    P_h = (Wq_h_all @ cov_full) @ Wq_h_all.transpose(-1, -2)       # (H, h_d, h_d)
    Q_h = (Wk_h_all @ cov_full) @ Wk_h_all.transpose(-1, -2)
    PQ = P_h @ Q_h                                                  # (H, h_d, h_d)
    # PQ is not symmetric in general; take the max |eigenvalue|.
    eigvals = torch.linalg.eigvals(PQ).abs()
    H_heads_t = eigvals.max(dim=-1).values.sqrt()
    H_heads = H_heads_t.detach().cpu().numpy().astype(np.float64)

    return H_agg, H_heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHECKPOINT_DIR)
    ap.add_argument("--dataset", default=DATASET_NAME)
    ap.add_argument("--results-dir", default=os.path.join(RESULTS_DIR, "horizon"))
    ap.add_argument("--n-examples", type=int, default=None,
                    help="Max examples (default: all long docs)")
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

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size

    # Force original_full attention
    backbone = _get_backbone(model)
    for layer in backbone.encoder.layer:
        layer.attention.self.attention_type = "original_full"
    model.config.attention_type = "original_full"

    print(f"  layers={num_layers}  heads={num_heads}  d={hidden_size}  "
          f"loaded in {time.time()-t0:.1f}s")

    # ── 2. Pre-extract Wq, Wk for all layers ────────────────────────────────
    print("[Step 2] Extracting attention weights")
    Wqs, Wks = [], []
    for l in range(num_layers):
        Wq, Wk = extract_qkv_weights(model, l)
        Wqs.append(Wq)
        Wks.append(Wk)

    # ── 3. Load + tokenize test set ──────────────────────────────────────────
    print(f"[Step 3] Loading dataset: {args.dataset}")
    t0 = time.time()

    if DATASET_LOCAL_DIR:
        ds = load_dataset("parquet", data_dir=DATASET_LOCAL_DIR, split="test")
    else:
        ds = load_dataset(args.dataset, split="test")

    def tokenize_fn(example):
        tok = tokenizer(
            example["text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "label": example["label"],
        }

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    ds.set_format("torch")

    # Keep only long documents
    lengths = np.array([int(ex["attention_mask"].sum()) for ex in ds])
    min_len = int(MAX_LENGTH * MIN_TOKEN_RATIO)
    long_idx = np.where(lengths >= min_len)[0]
    print(f"  docs>={min_len}: {len(long_idx)}/{len(ds)}")

    if args.n_examples is not None and len(long_idx) > args.n_examples:
        long_idx = long_idx[:args.n_examples]

    ds = ds.select(long_idx.tolist())

    # Distributed slicing
    end_idx = args.end_idx if args.end_idx is not None else len(ds)
    end_idx = min(end_idx, len(ds))
    if args.start_idx > 0 or args.end_idx is not None:
        ds = ds.select(list(range(args.start_idx, end_idx)))
        print(f"  Slice: [{args.start_idx}, {end_idx}) = {len(ds)} examples")

    n_ex = len(ds)
    print(f"  {n_ex} examples — {time.time()-t0:.1f}s")

    # ── Shard suffix ─────────────────────────────────────────────────────────
    shard = (f"_shard_{args.start_idx}_{end_idx}"
             if (args.start_idx > 0 or args.end_idx is not None) else "")

    # ── 4. Pre-allocate ──────────────────────────────────────────────────────
    H_agg   = np.zeros((n_ex, num_layers))
    H_heads = np.zeros((n_ex, num_layers, num_heads))
    lam_max     = np.zeros((n_ex, num_layers))
    trace_sig   = np.zeros((n_ex, num_layers))
    actual_lens = np.zeros(n_ex, dtype=int)
    labels_arr  = np.zeros(n_ex, dtype=int)

    # ── 5. Main loop ─────────────────────────────────────────────────────────
    print(f"[Step 4] Computing H and Σ stats for {n_ex} examples × {num_layers} layers")
    t0 = time.time()

    with torch.no_grad():
        for j in tqdm(range(n_ex), desc="Examples"):
            ex = ds[j]
            input_ids = ex["input_ids"].unsqueeze(0).to(device)
            attn_mask = ex["attention_mask"].unsqueeze(0).to(device)
            actual_len = int(attn_mask.sum().item())

            actual_lens[j] = actual_len
            labels_arr[j] = int(ex["label"].item())

            # Forward pass with hidden states
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
            # hidden_states: tuple of (num_layers + 1) tensors
            # hidden_states[0] = embeddings = input to layer 0
            # hidden_states[l] = output of layer l-1 = input to layer l
            hidden_states = out.hidden_states

            for l in range(num_layers):
                X_l = hidden_states[l]  # (1, seq_len, d) — input to layer l

                cov, lm, tr, _ = compute_sigma_stats(X_l, actual_len)
                lam_max[j, l] = lm
                trace_sig[j, l] = tr

                H_agg, H_heads = compute_horizon(
                    cov, Wqs[l], Wks[l], num_heads
                )
                H_agg[j, l]      = H_agg
                H_heads[j, l, :] = H_heads

            # Free memory
            del hidden_states, out

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/n_ex:.2f}s/example)")

    # ── 6. Save ──────────────────────────────────────────────────────────────
    print(f"[Step 5] Saving to {args.results_dir}")

    np.save(os.path.join(args.results_dir, f"H_agg{shard}.npy"), H_agg)
    np.save(os.path.join(args.results_dir, f"H_heads{shard}.npy"), H_heads)
    np.save(os.path.join(args.results_dir, f"lambda_max{shard}.npy"), lam_max)
    np.save(os.path.join(args.results_dir, f"trace_sigma{shard}.npy"), trace_sig)
    np.save(os.path.join(args.results_dir, f"actual_lens{shard}.npy"), actual_lens)
    np.save(os.path.join(args.results_dir, f"labels{shard}.npy"), labels_arr)

    # Metadata
    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "n_examples": n_ex,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hidden_size": hidden_size,
        "max_length": MAX_LENGTH,
        "min_token_ratio": MIN_TOKEN_RATIO,
        "start_idx": args.start_idx,
        "end_idx": end_idx,
        "elapsed_s": elapsed,
        "device": str(device),
    }
    with open(os.path.join(args.results_dir, f"meta{shard}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
