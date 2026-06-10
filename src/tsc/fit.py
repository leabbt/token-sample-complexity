"""Weighted-least-squares power-law fit in log-log space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class WLSFit:
    slope: float
    intercept: float
    cov: np.ndarray            # 2x2 covariance for [intercept, slope]

    @property
    def slope_se(self) -> float:
        return float(np.sqrt(max(self.cov[1, 1], 0.0)))


def wls_fit_full(logx: np.ndarray, logy: np.ndarray, var_logy: np.ndarray) -> WLSFit:
    """Fit `logy = intercept + slope * logx` weighted by `1 / var_logy`."""
    var_logy = np.maximum(np.asarray(var_logy, dtype=float), 1e-18)
    w = 1.0 / var_logy
    X = np.column_stack((np.ones_like(logx), logx))
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X, XtW @ logy)
    resid = logy - X @ beta
    dof = max(logx.size - 2, 1)
    sigma2 = (w * resid**2).sum() / dof
    cov = np.linalg.inv(XtW @ X) * sigma2
    return WLSFit(float(beta[1]), float(beta[0]), cov)


def fit_convergence(
    n_values: Sequence[int] | np.ndarray,
    err_means: Sequence[float] | np.ndarray,
    err_stds: Sequence[float] | np.ndarray,
    k: int,
) -> WLSFit:
    """Fit `err_mean ~ n^slope` via WLS in log-log space.

    `k` is the number of Monte-Carlo replicates per sample size (used to convert
    the per-replicate std to a standard error of the mean).
    """
    n_values = np.asarray(n_values, dtype=float)
    err_means = np.asarray(err_means, dtype=float)
    err_stds = np.asarray(err_stds, dtype=float)

    se = err_stds / np.sqrt(max(k, 1))
    var_logy = (se / np.maximum(err_means, 1e-12)) ** 2
    log_n = np.log(n_values)
    log_e = np.log(np.maximum(err_means, 1e-12))
    return wls_fit_full(log_n, log_e, var_logy)
