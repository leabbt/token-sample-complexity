"""Slope-vs-horizon plot for the window experiment. Three panels (mean, cov, MSE)
showing the fitted exponent `|β|` versus the per-layer horizon
`H = ‖Σ^{1/2}A‖₂`. Every layer found under `--results-dir` is included."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tsc.plotting import apply_paper_style, save_figure, MEAN_COLOR, COV_COLOR, THEORY_COLOR


def _collect(results_dir: Path):
    rows = []
    for meta_path in results_dir.glob("**/meta.json"):
        with open(meta_path) as f:
            rows.append(json.load(f))
    return rows


def plot(results_dir: Path, out_path: Path, theory: bool = True):
    apply_paper_style()
    rows = _collect(results_dir)
    if not rows:
        raise SystemExit(f"No meta.json under {results_dir}")

    H = np.array([r["horizon"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, key, se_key, color, title, ylabel in [
        (axes[0], "slope_m",   "slope_m_se",   MEAN_COLOR, "Mean",       r"$|\beta_{\mathrm{mean}}|$"),
        (axes[1], "slope_c",   "slope_c_se",   COV_COLOR,  "Covariance", r"$|\beta_{\mathrm{cov}}|$"),
        (axes[2], "slope_mse", "slope_mse_se", "k",        "MSE",        r"$|\beta_{\mathrm{MSE}}|$"),
    ]:
        y = np.abs(np.array([r[key] for r in rows]))
        ye = np.array([r.get(se_key, 0.0) for r in rows])
        ax.errorbar(H, y, yerr=ye, fmt="o", color=color, ms=6)
        if theory:
            x = np.geomspace(max(H.min(), 1e-3), max(H.max(), 1.0) * 2, 400)
            ax.plot(x, 1.0 / (2.0 * (1.0 + x**2)), color=THEORY_COLOR, lw=1.8,
                    label=r"$\frac{1}{2(1+H^2)}$")
        ax.axhline(0.5, color="0.6", ls="--", lw=1, alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"Horizon $H$"); ax.set_ylabel(ylabel)
        ax.set_title(title)
        if theory:
            ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"saved {out_path.with_suffix('.png')}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("figures/bigbird_window_slopes_vs_H"))
    p.add_argument("--no-theory", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.results_dir, args.out, theory=not args.no_theory)
