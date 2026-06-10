"""BigBird Monte-Carlo iid convergence experiment.

Loads a HuggingFace BigBird model, extracts the layer-`L` pre-attention embeddings
on a long text sample, runs a Monte-Carlo sweep over the i.i.d. subsample size
`n`, and fits the convergence rate of the mean and covariance errors against the
full-sequence reference. Writes one CSV per layer.

Example
-------
    python experiments/bigbird_iid/run.py --quick                     # local smoke test
    python experiments/bigbird_iid/run.py --source wiki --language en \\
        --layer 0 --N 122 --n_min 1000 --n_max 3000 --nb_tot 12 --k 500
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tsc import (
    chunked_attention_memory_efficient,
    fit_convergence,
    get_layer0_embeddings,
    get_spec,
    load_model,
    load_text,
    load_tokenizer,
    sample_iid,
)


def _get_pre_attention_X(model, tokens, layer_id: int) -> torch.Tensor:
    """Hidden state right before the attention block at layer `layer_id`.

    For layer 0 this is the embedding output (no transformer block needed).
    For later layers we forward-pass once with a hook on the layer's self-attention.
    """
    if layer_id == 0:
        return get_layer0_embeddings(model, tokens)

    cache = {}

    def hook(_module, inputs):
        cache["x"] = inputs[0].detach()

    handle = model.encoder.layer[layer_id].attention.self.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(**tokens)
    finally:
        handle.remove()
    return cache["x"]


def _qkv(model, X: torch.Tensor, layer_id: int):
    attn = model.encoder.layer[layer_id].attention.self
    with torch.no_grad():
        return attn.query(X), attn.key(X), attn.value(X)


def _horizon(model, X: torch.Tensor, layer_id: int) -> float:
    """Two-sided horizon `H = ‖Lᵀ A L‖₂`.

    `A = Wkᵀ Wq`. `L` is the lower-triangular Cholesky factor of `Σ + 1e-6·I`,
    where `Σ` is the sample covariance of the layer-`layer_id` input tokens X
    (`torch.linalg.cholesky` returns `L` with `L Lᵀ = Σ`).
    """
    attn = model.encoder.layer[layer_id].attention.self
    A_mat = attn.key.weight.detach().T @ attn.query.weight.detach()
    X1 = X[0]
    centered = X1 - X1.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / X1.size(0)
    cov_reg = cov + 1e-6 * torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
    L = torch.linalg.cholesky(cov_reg)
    return float(torch.linalg.matrix_norm(L.T @ A_mat @ L, ord=2))


def _run_mc_sweep(
    model, X, layer_id: int, mean_ref, cov_ref,
    n_min: int, n_max: int, nb_tot: int, k: int,
    head_size: int, all_size: int,
) -> pd.DataFrame:
    n_values = np.geomspace(n_min, n_max, nb_tot).astype(int)
    rows = []
    for n in tqdm(n_values, desc="MC sweep"):
        mean_errs, cov_errs = [], []
        for _ in range(k):
            X_sub = sample_iid(X, int(n), replace=True)
            Q, K, V = _qkv(model, X_sub, layer_id)
            mean_est, cov_est = chunked_attention_memory_efficient(
                Q, K, V,
                attention_head_size=head_size, all_head_size=all_size,
                use_fp16=True,
            )
            mean_errs.append(float(torch.linalg.vector_norm(mean_est - mean_ref, ord=2, dim=-1).mean()))
            cov_errs.append(float(torch.linalg.matrix_norm(cov_est - cov_ref, ord="fro", dim=(-2, -1)).mean()))
        rows.append({
            "n": int(n),
            "mean_err_mean": float(np.mean(mean_errs)),
            "mean_err_std": float(np.std(mean_errs)),
            "cov_err_mean": float(np.mean(cov_errs)),
            "cov_err_std": float(np.std(cov_errs)),
        })
    return pd.DataFrame(rows)


def run_one_layer(
    *, model_key: str, layer_id: int, max_length: int,
    source: str, language: str,
    n_min: int, n_max: int, nb_tot: int, k: int,
    output_dir: Path,
):
    spec = get_spec(model_key)
    model = load_model(model_key, attention_type="block_sparse")
    tokenizer = load_tokenizer(model_key)

    text = load_text(max_length=max_length, source=source, language=language)
    tokens = tokenizer(text, return_tensors="pt", max_length=max_length,
                       truncation=True, padding="max_length")
    tokens = {kk: vv.to(model.device) for kk, vv in tokens.items()}

    X = _get_pre_attention_X(model, tokens, layer_id)
    Q, K, V = _qkv(model, X, layer_id)
    head_size = spec.hidden_size // spec.num_attention_heads
    mean_ref, cov_ref = chunked_attention_memory_efficient(
        Q, K, V,
        attention_head_size=head_size, all_head_size=spec.hidden_size,
        use_fp16=True,
    )
    horizon = _horizon(model, X, layer_id)

    df = _run_mc_sweep(model, X, layer_id, mean_ref, cov_ref,
                      n_min, n_max, nb_tot, k,
                      head_size=head_size, all_size=spec.hidden_size)

    fit_m = fit_convergence(df["n"], df["mean_err_mean"], df["mean_err_std"], k)
    fit_c = fit_convergence(df["n"], df["cov_err_mean"], df["cov_err_std"], k)

    out = output_dir / source / language / f"layer{layer_id}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "convergence_data.csv", index=False)

    meta = {
        "model": model_key, "source": source, "language": language,
        "layer_id": layer_id, "max_length": max_length,
        "n_min": n_min, "n_max": n_max, "nb_tot": nb_tot, "k": k,
        "horizon": horizon,
        "slope_m": fit_m.slope, "slope_m_se": fit_m.slope_se,
        "slope_c": fit_c.slope, "slope_c_se": fit_c.slope_se,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"layer {layer_id}: H={horizon:.3f}  slope_m={fit_m.slope:.3f}  slope_c={fit_c.slope:.3f}")
    return meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bigbird-base",
                   choices=["bigbird-base", "bigbird-large"])
    p.add_argument("--source", default="wiki")
    p.add_argument("--language", default="en")
    p.add_argument("--layer", type=int, default=0, help="single layer to run")
    p.add_argument("--all-layers", action="store_true",
                   help="run every layer 0..L-1")
    p.add_argument("--N", type=int, default=1,
                   help="multiplier for max_length (max_length = 4096 * N)")
    p.add_argument("--n_min", type=int, default=1000)
    p.add_argument("--n_max", type=int, default=3000)
    p.add_argument("--nb_tot", type=int, default=12)
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--output-dir", default="results/bigbird_iid")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke-test parameters; runs in a few minutes on CPU")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.N, args.n_min, args.n_max, args.nb_tot, args.k = 1, 100, 400, 4, 2

    max_length = 4096 * args.N
    spec = get_spec(args.model)
    if max_length > spec.max_position_embeddings:
        raise SystemExit(
            f"max_length={max_length} exceeds {args.model}'s "
            f"max_position_embeddings={spec.max_position_embeddings}"
        )
    output_dir = Path(args.output_dir)

    if args.all_layers:
        n_layers = 12 if args.model == "bigbird-base" else 24
        layers = list(range(n_layers))
    else:
        layers = [args.layer]

    for layer_id in layers:
        run_one_layer(
            model_key=args.model, layer_id=layer_id, max_length=max_length,
            source=args.source, language=args.language,
            n_min=args.n_min, n_max=args.n_max, nb_tot=args.nb_tot, k=args.k,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
