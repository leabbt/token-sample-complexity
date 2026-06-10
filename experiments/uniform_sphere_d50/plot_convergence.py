"""Convergence-curve plot for the uniform-sphere d=50 experiment.

Two panels: mean and covariance error vs `n`, log-log, one curve per scale
(coloured by horizon). All scales found under `--results-dir` are plotted."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from tsc.plotting import apply_paper_style, save_figure


def _collect(results_dir: Path):
    runs = []
    for meta_csv in sorted(results_dir.glob("*_meta.csv")):
        data_csv = meta_csv.with_name(meta_csv.name.replace("_meta.csv", ".csv"))
        if not data_csv.exists():
            continue
        meta = pd.read_csv(meta_csv).iloc[0].to_dict()
        df = pd.read_csv(data_csv)
        runs.append((meta, df))
    runs.sort(key=lambda x: x[0]["horizon"])
    return runs


def plot(results_dir: Path, out_path: Path):
    apply_paper_style()
    runs = _collect(results_dir)
    if not runs:
        raise SystemExit(f"No results under {results_dir}")

    cmap = LinearSegmentedColormap.from_list("h", ["#B81D24", "#D97030", "#FFAB20"],
                                             N=max(len(runs), 2))

    fig, (ax_m, ax_c) = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, (meta, df) in enumerate(runs):
        color = cmap(i / max(len(runs) - 1, 1))
        n = df["n"].to_numpy()
        H = meta["horizon"]
        label = f"H={H:.2f}"
        if "mean_err_mean" in df.columns:
            ax_m.errorbar(n, df["mean_err_mean"], yerr=df["mean_err_std"],
                          fmt="o-", color=color, ms=4, label=label)
        if "cov_err_mean" in df.columns:
            ax_c.errorbar(n, df["cov_err_mean"], yerr=df["cov_err_std"],
                          fmt="s-", color=color, ms=4, label=label)

    for ax, title, ylabel in [
        (ax_m, "Mean error",       r"$\|\hat\mu - \mu_\infty\|_2$"),
        (ax_c, "Covariance error", r"$\|\hat\Sigma - \Sigma_\infty\|_2$"),
    ]:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$n$"); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9, loc="best", frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"saved {out_path.with_suffix('.png')}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("figures/uniform_sphere_d50_convergence"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.results_dir, args.out)
