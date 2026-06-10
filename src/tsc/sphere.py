"""Uniform-on-sphere experiment helpers: sampling, attention, closed-form reference
covariance, and the full Monte-Carlo sweep used by the d=50 experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ._sphere_covariance import exact_cov


def sample_sphere(n: int, d: int, radius: float, device, dtype=torch.float16) -> torch.Tensor:
    X = torch.randn(1, n, d, device=device, dtype=dtype)
    X = X / X.norm(dim=-1, keepdim=True)
    return radius * X


def sample_gaussian(n: int, d: int, sigma: float, device, dtype=torch.float16) -> torch.Tensor:
    return torch.randn(1, n, d, device=device, dtype=dtype) * sigma


def project_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    return X @ W_q.T, X @ W_k.T, X @ W_v.T


def attention_mean(Q, K, V, chunk_size: int = 8192) -> torch.Tensor:
    """Mean of attention output, computed chunk by chunk via fused SDPA."""
    batch, n, _ = Q.shape
    sum_y = torch.zeros(batch, Q.shape[-1], device=Q.device, dtype=Q.dtype)
    for q_start in range(0, n, chunk_size):
        q_end = min(q_start + chunk_size, n)
        y_chunk = F.scaled_dot_product_attention(Q[:, q_start:q_end, :], K, V)
        sum_y += y_chunk.sum(dim=1)
    return sum_y / n


def attention_mean_cov(Q, K, V, chunk_size: int = 512):
    """Online-softmax attention returning both mean and covariance of the output."""
    batch_size, seq_len, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    sum_y = torch.zeros(batch_size, d, device=Q.device, dtype=torch.float32)
    sum_yy = torch.zeros(batch_size, d, d, device=Q.device, dtype=torch.float32)
    num_q_chunks = (seq_len + chunk_size - 1) // chunk_size
    for qi in range(num_q_chunks):
        qs, qe = qi * chunk_size, min((qi + 1) * chunk_size, seq_len)
        q_len = qe - qs
        Q_chunk = Q[:, qs:qe, :]
        m = torch.full((batch_size, q_len, 1), float("-inf"), device=Q.device)
        l = torch.zeros(batch_size, q_len, 1, device=Q.device)
        o = torch.zeros(batch_size, q_len, d, device=Q.device)
        for ki in range(num_q_chunks):
            ks, ke = ki * chunk_size, min((ki + 1) * chunk_size, seq_len)
            K_chunk = K[:, ks:ke, :]
            V_chunk = V[:, ks:ke, :]
            scores = torch.matmul(Q_chunk, K_chunk.transpose(-1, -2)) * scale
            m_chunk = scores.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_chunk)
            exp_diff = torch.exp(m - m_new)
            l = l * exp_diff
            o = o * exp_diff
            exp_scores = torch.exp(scores - m_new)
            l = l + exp_scores.sum(dim=-1, keepdim=True)
            o = o + torch.matmul(exp_scores, V_chunk)
            m = m_new
        y_chunk = o / l
        sum_y += y_chunk.sum(dim=1)
        sum_yy += y_chunk.transpose(1, 2) @ y_chunk
    mean = sum_y / seq_len
    cov = (sum_yy / seq_len) - mean.unsqueeze(2) @ mean.unsqueeze(1)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    return mean, cov


def metric_error_mean(mean, mean_ref) -> float:
    return torch.linalg.vector_norm(mean - mean_ref, ord=2, dim=-1).mean().item()


def metric_error_cov(cov_emp, cov_ref) -> float:
    return torch.linalg.norm(cov_emp - cov_ref, ord=2).item()


@dataclass
class SphereConfig:
    d: int
    n_min: int
    n_max: int
    nb_points: int
    k: int
    W_q_scale: float
    W_k_scale: float
    W_v_scale: float = 1.0
    W_q_Id: int = 0       # 1 → identity * scale, 0 → random Gaussian * scale
    W_k_Id: int = 0
    W_v_Id: int = 0
    dist: str = "sphere"
    radius: float = 1.0
    chunk_size: int = 512
    mode: str = "both"    # "mean", "cov", "both"
    seed: int = 42


def _build_weight(d: int, scale: float, identity: bool, device, dtype):
    base = torch.eye(d) if identity else torch.randn(d, d)
    return (base * scale).to(device=device, dtype=dtype)


def run_sphere_experiment(cfg: SphereConfig, *, device=None, dtype=torch.float32):
    """Run the full Monte-Carlo sweep and return (results dataframe, metadata dict)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    W_q = _build_weight(cfg.d, cfg.W_q_scale, bool(cfg.W_q_Id), device, dtype)
    W_k = _build_weight(cfg.d, cfg.W_k_scale, bool(cfg.W_k_Id), device, dtype)
    W_v = _build_weight(cfg.d, cfg.W_v_scale, bool(cfg.W_v_Id), device, dtype)

    # Two-sided horizon  H = ‖Lᵀ A L‖₂  with  L Lᵀ = Σ.
    # For X uniform on S^{d-1}(R),  Σ = (R²/d)·I,  so L = (R/√d)·I (trivial).
    A_mat = W_k.T @ W_q
    cov_uniform = (cfg.radius ** 2 / cfg.d) * torch.eye(cfg.d, device=device, dtype=dtype)
    L = torch.linalg.cholesky(cov_uniform + 1e-12 * torch.eye(cfg.d, device=device, dtype=dtype))
    horizon = float(torch.linalg.matrix_norm(L.T @ A_mat @ L, ord=2))

    ref_mean = torch.zeros(cfg.d, device=device, dtype=dtype)
    ref_cov = None
    if cfg.mode in ("cov", "both"):
        ref_cov_np = exact_cov(
            W_q.cpu().numpy(), W_k.cpu().numpy(), W_v.cpu().numpy(),
            cfg.radius, cfg.d,
        )
        ref_cov = torch.tensor(ref_cov_np, dtype=dtype, device=device)

    n_values = np.geomspace(cfg.n_min, cfg.n_max, cfg.nb_points).astype(int)
    records = []
    sampler = sample_sphere if cfg.dist == "sphere" else sample_gaussian

    for n in tqdm(n_values, desc=f"sphere d={cfg.d}"):
        mean_errs, cov_errs = [], []
        for _ in range(cfg.k):
            X = sampler(int(n), cfg.d, cfg.radius, device, dtype).to(dtype)
            Q, K, V = project_qkv(X, W_q, W_k, W_v)
            if cfg.mode in ("cov", "both"):
                mean_est, cov_est = attention_mean_cov(Q, K, V, chunk_size=cfg.chunk_size)
                if cfg.mode == "both":
                    mean_errs.append(metric_error_mean(mean_est, ref_mean))
                cov_errs.append(metric_error_cov(cov_est.squeeze(0), ref_cov))
            else:
                mean_est = attention_mean(Q, K, V, chunk_size=cfg.chunk_size)
                mean_errs.append(metric_error_mean(mean_est, ref_mean))
        row = {"n": int(n)}
        if mean_errs:
            row["mean_err_mean"] = float(np.mean(mean_errs))
            row["mean_err_std"] = float(np.std(mean_errs) / math.sqrt(cfg.k))
        if cov_errs:
            row["cov_err_mean"] = float(np.mean(cov_errs))
            row["cov_err_std"] = float(np.std(cov_errs) / math.sqrt(cfg.k))
        records.append(row)

    df = pd.DataFrame(records)
    meta = {"d": cfg.d, "radius": cfg.radius, "horizon": horizon, "k": cfg.k,
            "dist": cfg.dist, "n_min": cfg.n_min, "n_max": cfg.n_max,
            "nb_points": cfg.nb_points, "mode": cfg.mode,
            "W_q_scale": cfg.W_q_scale, "W_k_scale": cfg.W_k_scale,
            "W_v_scale": cfg.W_v_scale, "seed": cfg.seed}

    log_n = np.log(df["n"].values)
    if "mean_err_mean" in df.columns:
        log_e = np.log(df["mean_err_mean"].values)
        (slope, _), cov_fit = np.polyfit(log_n, log_e, deg=1, cov=True)
        meta["mean_slope"] = float(slope)
        meta["mean_slope_std"] = float(np.sqrt(cov_fit[0, 0]))
    if "cov_err_mean" in df.columns:
        log_e = np.log(df["cov_err_mean"].values)
        (slope, _), cov_fit = np.polyfit(log_n, log_e, deg=1, cov=True)
        meta["cov_slope"] = float(slope)
        meta["cov_slope_std"] = float(np.sqrt(cov_fit[0, 0]))

    return df, meta
