"""Global downstream-classification experiment.

For each test example we run one dense forward pass to get the reference logits
and per-layer hidden states, then for every `n` in a log-spaced grid we run
`K_MC` sparse forward passes with `n` randomly sampled keys per layer. We record
per-layer mean / covariance / L2 errors against the dense reference, plus the
end-to-end logit error, KL divergence, accuracy and agreement rate.

Outputs `<results-dir>/metrics.json`, which feeds the camera-ready plot script.

Example
-------
    python experiments/downstream_global/run.py --quick
    python experiments/downstream_global/run.py \\
        --model <ckpt> --n-examples 500 --k-mc 20 \\
        --n-min 64 --n-max 4096 --nb-points 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import linregress
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model import BigBirdMeanPoolClassifier


def _generate_sparse_mask(seq_len, n, actual_len, seed, device):
    rng = torch.Generator().manual_seed(seed)
    n_sample = min(n, actual_len)
    perm = torch.randperm(actual_len, generator=rng)[:n_sample]
    mask = torch.full((1, 1, 1, seq_len), -10000.0, device=device)
    mask[0, 0, 0, perm.to(device)] = 0.0
    return mask


def _hook(n, seq_len, actual_len, seed, device):
    def fn(_module, args, kwargs):
        mask = _generate_sparse_mask(seq_len, n, actual_len, seed, device)
        if len(args) > 1 and args[1] is not None:
            return (args[0], args[1] + mask) + args[2:], kwargs
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            kw = dict(kwargs); kw["attention_mask"] = kw["attention_mask"] + mask
            return args, kw
        return args, kwargs
    return fn


def _install_hooks(model, n, seq_len, actual_len, base_seed, num_layers, device):
    backbone = model.bert
    handles = []
    for l in range(num_layers):
        attn = backbone.encoder.layer[l].attention.self
        handles.append(attn.register_forward_pre_hook(
            _hook(n, seq_len, actual_len, base_seed + l * 7919, device),
            with_kwargs=True,
        ))
    return handles


def _mean_err(h_s, h_d):
    return (h_s.mean(dim=1) - h_d.mean(dim=1)).norm().item()


def _cov_err(h_s, h_d):
    def _c(h):
        h = h.squeeze(0); h = h - h.mean(0, keepdim=True)
        return h.T @ h / max(h.shape[0] - 1, 1)
    return (_c(h_s) - _c(h_d)).norm().item()


def _l2_err(h_s, h_d):
    return (h_s - h_d).squeeze(0).pow(2).sum(dim=1).mean().item()


def _pool_l2_err(h_s, h_d, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    pool_s = (h_s * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    pool_d = (h_d * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return (pool_s - pool_d).norm().item()


def _fit_beta(n_values, errors):
    ok = errors > 1e-15
    if ok.sum() < 3:
        return {"beta": None, "R2": None, "std_err": None}
    log_n = np.log(n_values[ok].astype(float))
    log_e = np.log(errors[ok])
    slope, intercept, r, _, se = linregress(log_n, log_e)
    return {"beta": float(-slope), "R2": float(r**2),
            "std_err": float(se), "intercept": float(intercept)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=False, default=None,
                   help="HF model name or local path to fine-tuned checkpoint")
    p.add_argument("--dataset", default="ccdv/arxiv-classification")
    p.add_argument("--results-dir", type=Path, default=Path("results/downstream_global"))
    p.add_argument("--n-examples", type=int, default=500)
    p.add_argument("--k-mc", type=int, default=20)
    p.add_argument("--n-min", type=int, default=64)
    p.add_argument("--n-max", type=int, default=4096)
    p.add_argument("--nb-points", type=int, default=20)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--min-token-ratio", type=float, default=0.9)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_length = 512
        args.n_examples = 4
        args.k_mc = 2
        args.n_min, args.n_max, args.nb_points = 16, 256, 4
        args.min_token_ratio = 0.0
    if args.model is None:
        raise SystemExit("Provide --model (fine-tuned checkpoint dir or HF id).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  Model: {args.model}  Dataset: {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BigBirdMeanPoolClassifier.from_pretrained(args.model).to(device).eval()
    num_labels = model.config.num_labels
    num_layers = model.config.num_hidden_layers
    for layer in model.bert.encoder.layer:
        layer.attention.self.attention_type = "original_full"
    model.config.attention_type = "original_full"

    ds = load_dataset(args.dataset, split="test")

    def tokenize_fn(ex):
        tok = tokenizer(ex["text"], max_length=args.max_length,
                        truncation=True, padding="max_length")
        return {"input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "label": ex["label"]}

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
    ds.set_format("torch")
    lengths = np.array([int(ex["attention_mask"].sum()) for ex in ds])
    min_len = int(args.max_length * args.min_token_ratio)
    long_idx = np.where(lengths >= min_len)[0]
    sel = long_idx[: args.n_examples] if len(long_idx) >= args.n_examples \
        else np.arange(min(args.n_examples, len(ds)))
    ds = ds.select(sel.tolist())
    end_idx = args.end_idx if args.end_idx is not None else len(ds)
    end_idx = min(end_idx, len(ds))
    if args.start_idx > 0 or end_idx < len(ds):
        ds = ds.select(list(range(args.start_idx, end_idx)))
    n_ex = len(ds)
    print(f"  {n_ex} examples kept.")

    n_values = np.unique(np.append(
        np.geomspace(args.n_min, args.n_max, args.nb_points).astype(int),
        args.n_max,
    ))
    nb_n = len(n_values)
    print(f"  n grid ({nb_n}): {n_values.tolist()}")

    dense_logits = np.zeros((n_ex, num_labels))
    labels_arr = np.zeros(n_ex, dtype=int)
    sparse_logits = np.zeros((nb_n, n_ex, num_labels))
    err_mean = np.zeros((nb_n, n_ex, num_layers))
    err_cov = np.zeros((nb_n, n_ex, num_layers))
    err_l2 = np.zeros((nb_n, n_ex, num_layers))
    err_pool_l2 = np.zeros((nb_n, n_ex))

    t_infer = time.time()
    for i in tqdm(range(n_ex), desc="examples"):
        ids = ds[i]["input_ids"].unsqueeze(0).to(device)
        amsk = ds[i]["attention_mask"].unsqueeze(0).to(device)
        labels_arr[i] = ds[i]["label"].item()
        seq_len = ids.shape[1]
        actual_len = int(amsk.sum().item())

        with torch.no_grad():
            d_out = model(ids, attention_mask=amsk, output_hidden_states=True)
        dense_logits[i] = d_out.logits[0].cpu().numpy()
        d_hidden = [h.clone() for h in d_out.hidden_states[1:]]

        for j, n in enumerate(n_values):
            if n >= actual_len:
                sparse_logits[j, i] = dense_logits[i]
                continue
            logit_acc = np.zeros(num_labels)
            e_acc = np.zeros((3, num_layers))
            pool_acc = 0.0
            for mc in range(args.k_mc):
                seed = args.base_seed + i * 100_000 + j * 1_000 + mc
                hooks = _install_hooks(model, n, seq_len, actual_len, seed, num_layers, device)
                with torch.no_grad():
                    s_out = model(ids, attention_mask=amsk, output_hidden_states=True)
                for h in hooks:
                    h.remove()
                logit_acc += s_out.logits[0].cpu().numpy()
                for l in range(num_layers):
                    e_acc[0, l] += _mean_err(s_out.hidden_states[l + 1], d_hidden[l])
                    e_acc[1, l] += _cov_err(s_out.hidden_states[l + 1], d_hidden[l])
                    e_acc[2, l] += _l2_err(s_out.hidden_states[l + 1], d_hidden[l])
                pool_acc += _pool_l2_err(s_out.hidden_states[-1], d_hidden[-1], amsk)
            sparse_logits[j, i] = logit_acc / args.k_mc
            err_pool_l2[j, i] = pool_acc / args.k_mc
            err_mean[j, i] = e_acc[0] / args.k_mc
            err_cov[j, i] = e_acc[1] / args.k_mc
            err_l2[j, i] = e_acc[2] / args.k_mc
        del d_hidden
        if device.type == "cuda":
            torch.cuda.empty_cache()
    t_infer = time.time() - t_infer

    avg_mean = err_mean.mean(axis=1)
    avg_cov = err_cov.mean(axis=1)
    avg_l2 = err_l2.mean(axis=1)
    avg_pool_l2 = err_pool_l2.mean(axis=1)
    logit_err = np.array([
        np.linalg.norm(sparse_logits[j] - dense_logits, axis=1).mean()
        for j in range(nb_n)
    ])

    def _kl(p_log, q_log):
        p = torch.softmax(torch.tensor(p_log, dtype=torch.float64), 1).numpy()
        q = torch.softmax(torch.tensor(q_log, dtype=torch.float64), 1).numpy()
        p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
        return float(np.mean(np.sum(p * np.log(p / q), axis=1)))

    kl_vals = np.array([_kl(dense_logits, sparse_logits[j]) for j in range(nb_n)])

    dense_pred = dense_logits.argmax(1)
    acc_dense = np.array([(sparse_logits[j].argmax(1) == dense_pred).mean()
                          for j in range(nb_n)])
    acc_label = np.array([(sparse_logits[j].argmax(1) == labels_arr).mean()
                          for j in range(nb_n)])
    dense_accuracy = float((dense_pred == labels_arr).mean())

    fit_ok = n_values < args.n_max
    n_fit = n_values[fit_ok]
    beta = {f"layer_{l}_{m}": _fit_beta(n_fit, arr[fit_ok, l])
            for m, arr in (("mean", avg_mean), ("cov", avg_cov), ("l2", avg_l2))
            for l in range(num_layers)}
    beta["pool_l2"] = _fit_beta(n_fit, avg_pool_l2[fit_ok])
    beta["logit"] = _fit_beta(n_fit, logit_err[fit_ok])
    beta["kl"] = _fit_beta(n_fit, kl_vals[fit_ok])

    metrics = {
        "n_values": n_values.tolist(),
        "dense_accuracy": dense_accuracy,
        "logit_errors": logit_err.tolist(),
        "kl_divergences": kl_vals.tolist(),
        "accuracy_vs_dense": acc_dense.tolist(),
        "accuracy_vs_labels": acc_label.tolist(),
        "avg_layer_mean_errors": avg_mean.tolist(),
        "avg_layer_cov_errors": avg_cov.tolist(),
        "avg_layer_l2_errors": avg_l2.tolist(),
        "avg_pool_l2": avg_pool_l2.tolist(),
        "beta_fits": beta,
        "model_name": args.model, "dataset_name": args.dataset,
        "num_labels": num_labels, "num_layers": num_layers,
        "n_examples": n_ex, "k_mc": args.k_mc, "max_length": args.max_length,
        "inference_seconds": t_infer,
    }
    out = args.results_dir / "metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {out}.  Dense accuracy: {dense_accuracy:.4f}  ({t_infer:.0f}s)")


if __name__ == "__main__":
    main()
