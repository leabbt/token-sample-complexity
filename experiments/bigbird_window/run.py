"""BigBird Monte-Carlo window+random convergence experiment.

Same Monte-Carlo framework as `experiments/bigbird_iid/run.py`, but the
subsample is a centred window of half-width `w` plus `r` random tokens drawn
from outside the window. The effective sample size is `n_eff = 2w+1 + r`. We
sweep `w` (the window's half-width) and record three error metrics:
    - mean error  : ‖μ̂ - μ_full‖₂
    - cov error   : ‖Σ̂ - Σ_full‖_F
    - MSE         : (1/N) Σ_i ‖Y_full[i] - Y_sub[i]‖²

Example
-------
    python experiments/bigbird_window/run.py --quick
    python experiments/bigbird_window/run.py --source wiki --language en \\
        --layer 0 --N 122 --w_min 500 --w_max 1500 --nb_tot 12 --k 500 --random-tokens 50
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
    cross_attention_chunked,
    fit_convergence,
    full_attention_output_chunked,
    get_layer0_embeddings,
    get_spec,
    load_model,
    load_text,
    load_tokenizer,
    sample_window_and_random,
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


def _run_window_sweep(
    model, X, Q_full, V_full, K_full, Y_full, mean_ref, cov_ref,
    layer_id: int, w_min: int, w_max: int, nb_tot: int, k: int,
    random_tokens: int, head_size: int, all_size: int,
) -> pd.DataFrame:
    N = X.size(1)
    w_values = np.geomspace(w_min, w_max, nb_tot).astype(int)
    rows = []
    for w in tqdm(w_values, desc="window sweep"):
        mean_errs, cov_errs, mse_errs = [], [], []
        n_eff_obs = None
        for _ in range(k):
            X_sub = sample_window_and_random(X, int(w), random_tokens, query_pos=N // 2)
            if n_eff_obs is None:
                n_eff_obs = X_sub.size(1)
            Q_s, K_s, V_s = _qkv(model, X_sub, layer_id)

            mean_est, cov_est = chunked_attention_memory_efficient(
                Q_s, K_s, V_s,
                attention_head_size=head_size, all_head_size=all_size,
                use_fp16=True,
            )
            mean_errs.append(float(torch.linalg.vector_norm(mean_est - mean_ref, ord=2, dim=-1).mean()))
            cov_errs.append(float(torch.linalg.matrix_norm(cov_est - cov_ref, ord="fro", dim=(-2, -1)).mean()))

            Y_sub = cross_attention_chunked(
                Q_full, K_s, V_s,
                attention_head_size=head_size, all_head_size=all_size,
            )
            mse_errs.append(float(((Y_full - Y_sub.float()) ** 2).mean()))

        rows.append({
            "w": int(w), "n_eff": int(n_eff_obs),
            "mean_err_mean": float(np.mean(mean_errs)),
            "mean_err_std": float(np.std(mean_errs)),
            "cov_err_mean": float(np.mean(cov_errs)),
            "cov_err_std": float(np.std(cov_errs)),
            "mse_mean": float(np.mean(mse_errs)),
            "mse_std": float(np.std(mse_errs)),
        })
    return pd.DataFrame(rows)


def run_one_layer(
    *, model_key: str, layer_id: int, max_length: int,
    source: str, language: str,
    w_min: int, w_max: int, nb_tot: int, k: int, random_tokens: int,
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
    Q_full, K_full, V_full = _qkv(model, X, layer_id)
    head_size = spec.hidden_size // spec.num_attention_heads
    mean_ref, cov_ref = chunked_attention_memory_efficient(
        Q_full, K_full, V_full,
        attention_head_size=head_size, all_head_size=spec.hidden_size, use_fp16=True,
    )
    Y_full = full_attention_output_chunked(
        Q_full, K_full, V_full,
        attention_head_size=head_size, all_head_size=spec.hidden_size,
    )
    horizon = _horizon(model, X, layer_id)

    df = _run_window_sweep(
        model, X, Q_full, V_full, K_full, Y_full, mean_ref, cov_ref,
        layer_id, w_min, w_max, nb_tot, k, random_tokens,
        head_size=head_size, all_size=spec.hidden_size,
    )

    fit_m = fit_convergence(df["n_eff"], df["mean_err_mean"], df["mean_err_std"], k)
    fit_c = fit_convergence(df["n_eff"], df["cov_err_mean"], df["cov_err_std"], k)
    fit_mse = fit_convergence(df["n_eff"], df["mse_mean"], df["mse_std"], k)

    out = output_dir / source / language / f"layer{layer_id}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "convergence_data.csv", index=False)
    meta = {
        "model": model_key, "source": source, "language": language,
        "layer_id": layer_id, "max_length": max_length,
        "w_min": w_min, "w_max": w_max, "nb_tot": nb_tot, "k": k,
        "random_tokens": random_tokens, "horizon": horizon,
        "slope_m": fit_m.slope, "slope_m_se": fit_m.slope_se,
        "slope_c": fit_c.slope, "slope_c_se": fit_c.slope_se,
        "slope_mse": fit_mse.slope, "slope_mse_se": fit_mse.slope_se,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"layer {layer_id}: H={horizon:.3f}  βm={fit_m.slope:.3f}  βc={fit_c.slope:.3f}  βmse={fit_mse.slope:.3f}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bigbird-base",
                   choices=["bigbird-base", "bigbird-large"])
    p.add_argument("--source", default="wiki")
    p.add_argument("--language", default="en")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--N", type=int, default=1)
    p.add_argument("--w_min", type=int, default=500)
    p.add_argument("--w_max", type=int, default=1500)
    p.add_argument("--nb_tot", type=int, default=12)
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--random-tokens", type=int, default=50)
    p.add_argument("--output-dir", default="results/bigbird_window")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.N = 1
        args.w_min, args.w_max, args.nb_tot, args.k = 50, 200, 4, 2

    max_length = 4096 * args.N
    spec = get_spec(args.model)
    if max_length > spec.max_position_embeddings:
        raise SystemExit(
            f"max_length={max_length} exceeds {args.model}'s "
            f"max_position_embeddings={spec.max_position_embeddings}"
        )
    output_dir = Path(args.output_dir)
    layers = list(range(12 if args.model == "bigbird-base" else 24)) if args.all_layers else [args.layer]

    for layer_id in layers:
        run_one_layer(
            model_key=args.model, layer_id=layer_id, max_length=max_length,
            source=args.source, language=args.language,
            w_min=args.w_min, w_max=args.w_max, nb_tot=args.nb_tot, k=args.k,
            random_tokens=args.random_tokens, output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
