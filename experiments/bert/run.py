"""BERT Monte-Carlo iid convergence experiment.

Identical Monte-Carlo logic to the BigBird iid runner; only the model class is
different (`bert-base-uncased` / `bert-large-uncased`). The unified library in
`src/tsc/` handles both families through `load_model(model_key)`.

Example
-------
    python experiments/bert/run.py --quick
    python experiments/bert/run.py --source wiki --language en \\
        --layer 0 --n_min 64 --n_max 256 --nb_tot 8 --k 500
"""
from __future__ import annotations

import argparse
import json
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
    """Two-sided horizon `H = ‖Lᵀ A L‖₂` where `L Lᵀ = Σ + 1e-6·I` is the Cholesky
    factor of the sample covariance of `X`, and `A = Wkᵀ Wq`."""
    attn = model.encoder.layer[layer_id].attention.self
    A_mat = attn.key.weight.detach().T @ attn.query.weight.detach()
    X1 = X[0]
    centered = X1 - X1.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / X1.size(0)
    cov_reg = cov + 1e-6 * torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
    L = torch.linalg.cholesky(cov_reg)
    return float(torch.linalg.matrix_norm(L.T @ A_mat @ L, ord=2))


def _run_mc_sweep(model, X, layer_id, mean_ref, cov_ref,
                  n_min, n_max, nb_tot, k, head_size, all_size):
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


def run_one_layer(*, model_key, layer_id, max_length, source, language,
                  n_min, n_max, nb_tot, k, output_dir):
    spec = get_spec(model_key)
    model = load_model(model_key)
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
        attention_head_size=head_size, all_head_size=spec.hidden_size, use_fp16=True,
    )
    horizon = _horizon(model, X, layer_id)

    df = _run_mc_sweep(model, X, layer_id, mean_ref, cov_ref,
                      n_min, n_max, nb_tot, k, head_size, spec.hidden_size)
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
    print(f"layer {layer_id}: H={horizon:.3f}  βm={fit_m.slope:.3f}  βc={fit_c.slope:.3f}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bert-base",
                   choices=["bert-base", "bert-large"])
    p.add_argument("--source", default="wiki")
    p.add_argument("--language", default="en")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--max-length", type=int, default=512,
                   help="BERT max position embeddings is 512.")
    p.add_argument("--n_min", type=int, default=64)
    p.add_argument("--n_max", type=int, default=384)
    p.add_argument("--nb_tot", type=int, default=10)
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--output-dir", default="results/bert")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_length = 256
        args.n_min, args.n_max, args.nb_tot, args.k = 32, 128, 4, 2

    spec = get_spec(args.model)
    if args.max_length > spec.max_position_embeddings:
        raise SystemExit(
            f"max_length={args.max_length} exceeds {args.model}'s "
            f"max_position_embeddings={spec.max_position_embeddings}"
        )

    output_dir = Path(args.output_dir)
    layers = list(range(12 if args.model == "bert-base" else 24)) if args.all_layers else [args.layer]
    for layer_id in layers:
        run_one_layer(
            model_key=args.model, layer_id=layer_id, max_length=args.max_length,
            source=args.source, language=args.language,
            n_min=args.n_min, n_max=args.n_max, nb_tot=args.nb_tot, k=args.k,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
