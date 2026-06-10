"""Slope-vs-horizon plot for the uniform-sphere d=50 experiment.

All horizons found under `--results-dir` are plotted; no horizon cutoff is
applied (the `H_COV_CUTOFF` masking from the original codebase is not used)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsc.plotting import apply_paper_style, save_figure, MEAN_COLOR, COV_COLOR, THEORY_COLOR


def _collect(results_dir: Path):
    rows = []
    for meta_csv in results_dir.glob("*_meta.csv"):
        rows.append(pd.read_csv(meta_csv).iloc[0].to_dict())
    rows.sort(key=lambda r: r["horizon"])
    return rows


def plot(results_dir: Path, out_path: Path, theory: bool = True):
    apply_paper_style()
    rows = _collect(results_dir)
    if not rows:
        raise SystemExit(f"No *_meta.csv under {results_dir}")

    H = np.array([r["horizon"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    if "mean_slope" in rows[0]:
        sm = np.abs(np.array([r["mean_slope"] for r in rows]))
        ye = np.array([r.get("mean_slope_std", 0.0) for r in rows])
        ax.errorbar(H, sm, yerr=ye, fmt="o", color=MEAN_COLOR, ms=6,
                    label=r"$|\beta_{\mathrm{mean}}|$")
    if "cov_slope" in rows[0]:
        sc = np.abs(np.array([r["cov_slope"] for r in rows]))
        ye = np.array([r.get("cov_slope_std", 0.0) for r in rows])
        ax.errorbar(H, sc, yerr=ye, fmt="s", color=COV_COLOR, ms=6,
                    label=r"$|\beta_{\mathrm{cov}}|$")

    if theory:
        x = np.geomspace(max(H.min(), 1e-3), max(H.max(), 1.0) * 2, 500)
        ax.plot(x, 1.0 / (2.0 * (1.0 + x**2)), color=THEORY_COLOR, lw=2,
                label=r"$\frac{1}{2(1+H^2)}$")
    ax.axhline(0.5, color="0.6", ls="--", lw=1, alpha=0.8, label=r"$|\beta|=0.5$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Horizon $H$")
    ax.set_ylabel(r"$|\beta|$")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()

    save_figure(fig, out_path)
    plt.close(fig)
    print(f"saved {out_path.with_suffix('.png')}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("figures/uniform_sphere_d50_slopes_vs_H"))
    p.add_argument("--no-theory", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.results_dir, args.out, theory=not args.no_theory)
