"""Gaussian synthetic convergence figure: fitted convergence rates `|β_mean|`
and `|β_cov|` versus the horizon `H = ‖Σ^{1/2}AΣ^{1/2}‖₂`, with the theoretical
reference `1/(2(1+H²))`. Every point is plotted uniformly — no subset is
starred or otherwise singled out."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsc.plotting import apply_paper_style, save_figure, MEAN_COLOR, COV_COLOR, THEORY_COLOR


def plot(results_csv: Path, out_path: Path):
    apply_paper_style()
    df = pd.read_csv(results_csv).sort_values("horizon")
    H = df["horizon"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.errorbar(H, np.abs(df["slope_m"]), yerr=df["slope_m_se"],
                fmt="o", color=MEAN_COLOR, ms=6, label=r"$|\beta_{\mathrm{mean}}|$")
    ax.errorbar(H, np.abs(df["slope_c"]), yerr=df["slope_c_se"],
                fmt="s", color=COV_COLOR, ms=6, label=r"$|\beta_{\mathrm{cov}}|$")

    x = np.geomspace(max(H.min(), 1e-3), max(H.max(), 1.0) * 2, 500)
    ax.plot(x, 1.0 / (2.0 * (1.0 + x**2)), color=THEORY_COLOR, lw=2,
            label=r"$\frac{1}{2(1+H^2)}$")
    ax.axhline(0.5, color="0.6", ls="--", lw=1, alpha=0.8, label=r"$|\beta|=0.5$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Horizon $H = \|\Sigma^{1/2}A\Sigma^{1/2}\|_2$")
    ax.set_ylabel(r"$|\beta|$")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()

    save_figure(fig, out_path)
    plt.close(fig)
    print(f"saved {out_path.with_suffix('.png')}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/gaussian/results.csv"))
    p.add_argument("--out", type=Path, default=Path("figures/gaussian_convergence"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.results, args.out)
