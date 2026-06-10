"""Convergence-curve plot: mean and covariance error vs sample size `n`, log-log,
overlaid for every (layer, dataset) found under `--results-dir`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsc.plotting import apply_paper_style, save_figure, LAYER_CMAP, MEAN_COLOR, COV_COLOR


def _gather(results_dir: Path):
    """Yield (source, language, layer_id, df, meta) tuples for every result."""
    for csv in results_dir.glob("**/convergence_data.csv"):
        layer_dir = csv.parent
        layer_id = int(layer_dir.name.removeprefix("layer"))
        language = layer_dir.parent.name
        source = layer_dir.parent.parent.name
        with open(layer_dir / "meta.json") as f:
            meta = json.load(f)
        yield source, language, layer_id, pd.read_csv(csv), meta


def _plot_panel(ax, n, err, slope, intercept, color, label, marker):
    ax.errorbar(n, err, fmt=marker, color=color, ms=5, lw=1, label=label)
    n_fit = np.geomspace(n.min(), n.max(), 100)
    ax.plot(n_fit, np.exp(intercept) * n_fit ** slope, "--", color=color, lw=1, alpha=0.7)


def plot(results_dir: Path, out_path: Path):
    apply_paper_style()
    fig, (ax_m, ax_c) = plt.subplots(1, 2, figsize=(12, 4.6))
    runs = list(_gather(results_dir))
    if not runs:
        raise SystemExit(f"No convergence_data.csv under {results_dir}")

    for src, lang, layer_id, df, meta in runs:
        n = df["n"].to_numpy()
        color = LAYER_CMAP(layer_id / 11)
        label = f"{src}/{lang} L{layer_id}"
        slope_m = meta["slope_m"]
        slope_c = meta["slope_c"]
        intercept_m = np.log(df["mean_err_mean"].iloc[0]) - slope_m * np.log(n[0])
        intercept_c = np.log(df["cov_err_mean"].iloc[0]) - slope_c * np.log(n[0])
        _plot_panel(ax_m, n, df["mean_err_mean"].to_numpy(), slope_m, intercept_m,
                    color, label, "o")
        _plot_panel(ax_c, n, df["cov_err_mean"].to_numpy(), slope_c, intercept_c,
                    color, label, "s")

    for ax, ylabel, title in [
        (ax_m, r"$\|\hat\mu_n - \mu_\infty\|_2$", "Mean convergence"),
        (ax_c, r"$\|\hat\Sigma_n - \Sigma_\infty\|_F$", "Covariance convergence"),
    ]:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$n$"); ax.set_ylabel(ylabel)
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
    p.add_argument("--out", type=Path,
                   default=Path("figures/bigbird_iid_convergence"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.results_dir, args.out)
