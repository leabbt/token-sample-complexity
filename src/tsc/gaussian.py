"""Synthetic Gaussian-token experiment: Monte-Carlo convergence of mean and
covariance attention statistics swept over a horizon."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .fit import fit_convergence


def _attn(Q, K, V, tiling: bool = False) -> torch.Tensor:
    scale = 1.0 / math.sqrt(Q.size(-1))
    if tiling:
        return F.scaled_dot_product_attention(Q, K, V)
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    return torch.matmul(torch.softmax(scores, dim=-1), V)


def _mean_cov(x: torch.Tensor):
    n = x.size(1)
    mean = x.mean(dim=1)
    cov = (x.transpose(1, 2) @ x) / n - mean.unsqueeze(2) @ mean.unsqueeze(1)
    return mean, cov


def _sample(dist, n):
    return dist.sample((n,)).unsqueeze(0)


@dataclass
class GaussianConfig:
    d: int
    rho: float                  # Σ = rho * I_d
    n_reference: int            # tokens used to build the finite reference distribution
    n_min: int
    n_max: int
    nb_tot: int
    k: int
    scales: np.ndarray          # attention-scale grid (controls horizon)
    seed: int = 0
    tiling: bool = False        # use scaled_dot_product_attention when True


def _project(X, Wq, Wk, Wv):
    return X @ Wq.T, X @ Wk.T, X @ Wv.T


def run_gaussian_horizon_sweep(
    cfg: GaussianConfig,
    *,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Sweep `cfg.scales`, fit `error ~ n^slope` for mean and covariance at each scale.

    Returns a DataFrame with one row per scale and columns
    `scale, horizon, slope_m, slope_m_se, slope_c, slope_c_se`.
    `horizon = ‖Σ^{1/2} A Σ^{1/2}‖₂` with `A = K^T Q * scale` for a fixed
    random (Q, K, V).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    sigma = torch.eye(cfg.d, device=device) * cfg.rho
    # Two-sided horizon convention: use Cholesky factor (trivial for Σ = ρ I).
    sigma_chol_L = torch.linalg.cholesky(sigma + 1e-12 * torch.eye(cfg.d, device=device))
    dist = torch.distributions.MultivariateNormal(torch.zeros(cfg.d, device=device), sigma)

    weight_scale = 1.0 / math.sqrt(cfg.d)
    Wq_base = torch.randn(cfg.d, cfg.d, device=device) * weight_scale
    Wk_base = torch.randn(cfg.d, cfg.d, device=device) * weight_scale
    Wv = torch.randn(cfg.d, cfg.d, device=device) * weight_scale

    n_values = np.geomspace(cfg.n_min, cfg.n_max, cfg.nb_tot).astype(int)
    records = []

    for scale in tqdm(cfg.scales, desc=f"gaussian d={cfg.d}"):
        Wq = Wq_base * math.sqrt(scale)
        Wk = Wk_base * math.sqrt(scale)
        A = Wk.T @ Wq
        horizon = float(torch.linalg.matrix_norm(sigma_chol_L.T @ A @ sigma_chol_L, ord=2))

        ref_tokens = _sample(dist, cfg.n_reference)
        Qr, Kr, Vr = _project(ref_tokens, Wq, Wk, Wv)
        mean_ref, cov_ref = _mean_cov(_attn(Qr, Kr, Vr, cfg.tiling))

        mean_means, mean_stds, cov_means, cov_stds = [], [], [], []
        for n in n_values:
            mean_e, cov_e = [], []
            for _ in range(cfg.k):
                X = _sample(dist, int(n))
                Q, K, V = _project(X, Wq, Wk, Wv)
                mean_est, cov_est = _mean_cov(_attn(Q, K, V, cfg.tiling))
                mean_e.append(torch.linalg.vector_norm(mean_est - mean_ref, ord=2, dim=-1).mean().item())
                cov_e.append(torch.linalg.matrix_norm(cov_est - cov_ref, ord="fro", dim=(-2, -1)).mean().item())
            mean_means.append(float(np.mean(mean_e)))
            mean_stds.append(float(np.std(mean_e)))
            cov_means.append(float(np.mean(cov_e)))
            cov_stds.append(float(np.std(cov_e)))

        fit_m = fit_convergence(n_values, mean_means, mean_stds, cfg.k)
        fit_c = fit_convergence(n_values, cov_means, cov_stds, cfg.k)
        records.append({
            "scale": float(scale),
            "horizon": horizon,
            "slope_m": fit_m.slope,
            "slope_m_se": fit_m.slope_se,
            "slope_c": fit_c.slope,
            "slope_c_se": fit_c.slope_se,
        })

    return pd.DataFrame(records)
