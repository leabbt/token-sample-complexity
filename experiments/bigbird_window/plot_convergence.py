"""Convergence-curve plot for the window experiment: mean, cov, and MSE error
versus the effective sample size `n_eff = 2w+1 + r`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsc.plotting import apply_paper_style, save_figure, LAYER_CMAP


def _gather(results_dir: Path):
    for csv in results_dir.glob("**/convergence_data.csv"):
        layer_dir = csv.parent
        layer_id = int(layer_dir.name.removeprefix("layer"))
        language = layer_dir.parent.name
        source = layer_dir.parent.parent.name
        with open(layer_dir / "meta.json") as f:
            meta = json.load(f)
        yield source, language, layer_id, pd.read_csv(csv), meta


def plot(results_dir: Path, out_path: Path):
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    runs = list(_gather(results_dir))
    if not runs:
        raise SystemExit(f"No convergence_data.csv under {results_dir}")

    for src, lang, layer_id, df, meta in runs:
        n = df["n_eff"].to_numpy()
        color = LAYER_CMAP(layer_id / 11)
        label = f"{src}/{lang} L{layer_id}"
        for ax, col, slope_key in [
            (axes[0], "mean_err_mean", "slope_m"),
            (axes[1], "cov_err_mean",  "slope_c"),
            (axes[2], "mse_mean",      "slope_mse"),
        ]:
            err = df[col].to_numpy()
            slope = meta[slope_key]
            intercept = np.log(err[0]) - slope * np.log(n[0])
            ax.errorbar(n, err, fmt="o", color=color, ms=5, lw=1, label=label)
            ax.plot(np.geomspace(n.min(), n.max(), 100),
                    np.exp(intercept) * np.geomspace(n.min(), n.max(), 100) ** slope,
                    "--", color=color, lw=1, alpha=0.7)

    for ax, title, ylabel in [
        (axes[0], "Mean error",       r"$\|\hat\mu - \mu_\infty\|_2$"),
        (axes[1], "Covariance error", r"$\|\hat\Sigma - \Sigma_\infty\|_F$"),
        (axes[2], "MSE",              r"$\frac{1}{N}\sum_i \|Y_{\mathrm{full}}^{(i)} - Y_{\mathrm{sub}}^{(i)}\|^2$"),
    ]:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$n_{\mathrm{eff}}$"); ax.set_ylabel(ylabel)
        ax.set_title(title)
        if len(runs) <= 12:
            ax.legend(fontsize=8, loc="best", frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"saved {out_path.with_suffix('.png')}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("figures/bigbird_window_convergence"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.results_dir, args.out)
