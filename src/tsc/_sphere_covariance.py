"""
Exact covariance of the continuous attention operator on the sphere.

Gamma_A(x) = R * A_d(R ||Ax||) * Ax / ||Ax||,   A_d(r) = I_{d/2}(r) / I_{d/2-1}(r)
g(x) = W_v Gamma_A(x)

Cov(Gamma_A(X)) = U diag(lambda) U^T   where A = U diag(sigma) V^T
C_ref = W_v Cov(Gamma_A(X)) W_v^T
"""

import numpy as np
from scipy.special import ive, roots_jacobi
from scipy.special import beta as beta_func_scipy


def A_d(r, d):
    """Bessel ratio A_d(r) = I_{d/2}(r) / I_{d/2-1}(r) via exponentially scaled Bessel.

    Uses ive to avoid overflow. Returns 0 when r=0 (by continuity A_d(0)=0).
    """
    r = np.asarray(r, dtype=float)
    scalar = r.ndim == 0
    r = np.atleast_1d(r)
    nu_num = d / 2.0
    nu_den = d / 2.0 - 1.0
    out = np.zeros_like(r)
    mask = r > 0
    if mask.any():
        den = ive(nu_den, r[mask])
        safe = np.abs(den) > 1e-300
        out_vals = np.zeros(mask.sum())
        out_vals[safe] = ive(nu_num, r[mask][safe]) / den[safe]
        out[mask] = out_vals
    return float(out[0]) if scalar else out


def _integrand_ratio(s2, R, d):
    """Compute A_d(R^2 sqrt(s2))^2 / s2, handling s2->0 by continuity.

    As r->0, A_d(r) ~ r/d, so A_d(R^2 sqrt(s2))^2 / s2 -> R^4 / d^2.
    """
    s2 = np.asarray(s2, dtype=float)
    scalar = s2.ndim == 0
    s2 = np.atleast_1d(s2)
    out = np.full_like(s2, R**4 / d**2)
    mask = s2 > 1e-30
    if mask.any():
        r_arg = R**2 * np.sqrt(s2[mask])
        Ad_val = A_d(r_arg, d)
        out[mask] = Ad_val**2 / s2[mask]
    return float(out[0]) if scalar else out


def _stick_breaking_to_simplex_batch(v_all):
    """Convert (N, d-1) stick-breaking variables to (N, d) simplex points."""
    N, dm1 = v_all.shape
    d = dm1 + 1
    t = np.zeros((N, d))
    remaining = np.ones(N)
    for k in range(dm1):
        t[:, k] = v_all[:, k] * remaining
        remaining *= (1.0 - v_all[:, k])
    t[:, d - 1] = remaining
    return t


def lambda_i_from_quadrature(sigmas, R, d, n_quad=None):
    """Compute all eigenvalues lambda_i of Cov(Gamma_A(X)) via Gauss-Jacobi quadrature.

    Uses stick-breaking parameterization of Dir(1/2,...,1/2):
      v_k ~ Beta(1/2, (d-k-1)/2),  k = 0,...,d-2
    and tensor-product Gauss-Jacobi quadrature (one rule per v_k).

    The integral is E_T[ A_d(R^2 sqrt(sum_k sigma_k^2 T_k))^2 / (sum_k sigma_k^2 T_k) * T_i ]
    with T ~ Dir(1/2,...,1/2).

    Returns lambda_i = R^2 sigma_i^2 * <that expectation>.
    If sigma_i = 0, then lambda_i = 0 (short-circuited).
    """
    sigmas = np.asarray(sigmas, dtype=float)
    sig2 = sigmas**2

    if d == 1:
        if sig2[0] == 0:
            return np.array([0.0])
        Ad_val = A_d(R**2 * np.sqrt(sig2[0]), d)
        return np.array([R**2 * Ad_val**2])

    # Auto-select n_quad: keep grid size under MAX_GRID
    MAX_GRID = 5_000_000
    if n_quad is None:
        n_quad = max(3, int(MAX_GRID ** (1.0 / (d - 1))))
        n_quad = min(n_quad, 50)

    total_pts = n_quad ** (d - 1)
    if total_pts > 1e9:
        raise ValueError(
            f"Tensor-product quadrature infeasible for d={d} "
            f"(would need {total_pts:.0e} grid points). "
            f"Reduce n_quad or use d <= ~12."
        )

    # Build 1D Gauss-Jacobi rules for each stick-breaking variable.
    # v_k ~ Beta(a, b) with a=1/2, b=(d-k-1)/2.
    # On [-1,1]: weight (1-x)^alpha (1+x)^beta  with alpha=b-1, beta=a-1.
    # Map to [0,1]: v = (1+x)/2, integral picks up factor 2^{-(a+b-1)} / B(a,b).
    nodes_list = []
    weights_list = []
    for k in range(d - 1):
        a_b = 0.5
        b_b = (d - k - 1) / 2.0
        alpha_j = b_b - 1.0
        beta_j = a_b - 1.0   # = -0.5
        xj, wj = roots_jacobi(n_quad, alpha_j, beta_j)
        vj = (1.0 + xj) / 2.0
        scale = 2.0**(-(a_b + b_b - 1.0)) / beta_func_scipy(a_b, b_b)
        nodes_list.append(vj)
        weights_list.append(wj * scale)

    # Build tensor-product grid
    grids = np.meshgrid(
        *[np.arange(n_quad) for _ in range(d - 1)], indexing="ij"
    )
    idx = np.stack([g.ravel() for g in grids], axis=-1)  # (N, d-1)
    N = idx.shape[0]

    v_all = np.empty((N, d - 1))
    w_all = np.ones(N)
    for k in range(d - 1):
        v_all[:, k] = nodes_list[k][idx[:, k]]
        w_all *= weights_list[k][idx[:, k]]

    # Stick-breaking -> simplex
    t_all = _stick_breaking_to_simplex_batch(v_all)

    # s2 = sum_k sigma_k^2 t_k
    s2_all = t_all @ sig2

    # A_d(R^2 sqrt(s2))^2 / s2, with correct limit at s2=0
    ratio_all = _integrand_ratio(s2_all, R, d)

    # common = ratio * quadrature_weight
    common = ratio_all * w_all

    # lambda_i = R^2 sigma_i^2 * E[ratio * T_i]
    # Short-circuit sigma_i = 0
    expectations = (t_all * common[:, None]).sum(axis=0)
    lambdas = R**2 * sig2 * expectations
    lambdas[sig2 == 0] = 0.0

    return lambdas


def lambda_i_from_dirichlet_mc(sigmas, R, d, n_samples=1_000_000, seed=0):
    """Compute eigenvalues lambda_i by Monte Carlo sampling from Dir(1/2,...,1/2).

    Fallback for large d where tensor-product quadrature is infeasible.
    Still evaluates the exact closed-form formula — only the integral is approximated.

    Args:
        sigmas: singular values of A (length d)
        R: sphere radius
        d: dimension
        n_samples: number of Dirichlet samples
        seed: random seed for reproducibility

    Returns:
        lambda_i array of length d
    """
    sigmas = np.asarray(sigmas, dtype=float)
    sig2 = sigmas**2

    rng = np.random.default_rng(seed)
    # Dir(1/2,...,1/2) via Gamma(1/2,1) samples
    alpha = np.full(d, 0.5)
    G = rng.gamma(alpha, size=(n_samples, d))
    T = G / G.sum(axis=1, keepdims=True)  # (n_samples, d)

    # s2 = sum_k sigma_k^2 T_k
    s2 = T @ sig2  # (n_samples,)

    # A_d(R^2 sqrt(s2))^2 / s2, with correct limit at s2=0
    ratio = _integrand_ratio(s2, R, d)  # (n_samples,)

    # E[ratio * T_i] for each i
    expectations = (T * ratio[:, None]).mean(axis=0)
    lambdas = R**2 * sig2 * expectations
    lambdas[sig2 == 0] = 0.0

    return lambdas


# Threshold: use quadrature for d <= MAX_D_QUADRATURE, MC above
MAX_D_QUADRATURE = 12


def exact_cov_gamma(A, R, d, n_quad=None, n_mc_samples=1_000_000):
    """Compute Cov(Gamma_A(X)) = U diag(lambda) U^T from matrix A.

    For d <= 12, uses Gauss-Jacobi quadrature (deterministic).
    For d > 12, uses Monte Carlo on Dir(1/2,...,1/2) to evaluate the exact formula.

    Args:
        A: d x d numpy array
        R: sphere radius
        d: dimension
        n_quad: quadrature points per dimension (auto if None, only for d <= 12)
        n_mc_samples: number of Dirichlet MC samples (only for d > 12)

    Returns:
        Cov(Gamma_A(X)), d x d numpy array
    """
    U, sigmas, _ = np.linalg.svd(A)

    if d <= MAX_D_QUADRATURE:
        lambdas = lambda_i_from_quadrature(sigmas, R, d, n_quad=n_quad)
    else:
        print(f"  d={d} > {MAX_D_QUADRATURE}: using Dirichlet MC "
              f"({n_mc_samples:.0e} samples) for exact formula integration")
        lambdas = lambda_i_from_dirichlet_mc(sigmas, R, d, n_samples=n_mc_samples)

    cov_gamma = (U * lambdas[None, :]) @ U.T
    return 0.5 * (cov_gamma + cov_gamma.T)


def exact_cov(W_q, W_k, W_v, R, d, n_quad=None, n_mc_samples=1_000_000):
    """Compute C_ref = W_v Cov(Gamma_A(X)) W_v^T  with  A = W_k^T W_q / sqrt(d).

    Args:
        W_q, W_k, W_v: d x d numpy arrays
        R: sphere radius
        d: dimension
        n_quad: quadrature points per dimension (auto if None, only for small d)
        n_mc_samples: number of Dirichlet MC samples (only for large d)

    Returns:
        C_ref, d x d numpy array
    """
    W_q = np.asarray(W_q, dtype=float)
    W_k = np.asarray(W_k, dtype=float)
    W_v = np.asarray(W_v, dtype=float)

    A = W_k.T @ W_q / np.sqrt(d)
    cov_gamma = exact_cov_gamma(A, R, d, n_quad=n_quad, n_mc_samples=n_mc_samples)
    C_ref = W_v @ cov_gamma @ W_v.T
    return 0.5 * (C_ref + C_ref.T)
