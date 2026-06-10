"""Camera-ready downstream-task figures.

Two panels (each saved separately to a single PNG/PDF in the chosen ICML size):

  1. classification_errors  — RMSE, Pooling, Logit, Disagreement vs n. Each
     metric is normalised to its first point so all curves coincide at n_min,
     fit a pinned no-intercept slope, and have a tapered confidence band.
  2. accuracy_agreement     — accuracy vs labels, agreement rate vs dense
     predictions, and a horizontal reference at the dense accuracy.

Reads `metrics.json` produced by `run.py`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tsc.plotting import apply_paper_style, save_figure

C_RMSE   = "#B81D24"
C_POOL   = "#FFAB20"
C_LOGIT  = "#2CA02C"
C_DISAGR = "#2166AC"
C_ACC    = "#7E2F8E"
C_AGR    = "#1F77B4"


def _load(metrics_path: Path):
    with open(metrics_path) as f:
        m = json.load(f)
    n = np.array(m["n_values"], dtype=float)
    return {
        "n":            n,
        "rmse":         np.array(m["avg_layer_mean_errors"]).mean(axis=1),
        "pool":         np.array(m["avg_pool_l2"]),
        "logit":        np.array(m["logit_errors"]),
        "acc_dense":    float(m["dense_accuracy"]),
        "acc_labels":   np.array(m["accuracy_vs_labels"]),
        "acc_vs_dense": np.array(m["accuracy_vs_dense"]),
        "n_examples":   int(m["n_examples"]),
        "k_mc":         int(m["k_mc"]),
    }


def _fit_pinned(arr, fit_mask, n_full):
    """No-intercept regression on log(y/y0) vs log(n/n0). All curves pass
    through (n_min, 1). Returns (y_norm, slope, band)."""
    y = arr[fit_mask]
    y_norm = y / y[0]
    log_n = np.log(n_full[fit_mask] / n_full[fit_mask][0])
    log_y = np.log(y_norm)
    slope = float(np.sum(log_n * log_y) / np.sum(log_n ** 2))
    resid = log_y - slope * log_n
    return y_norm, slope, float(np.std(resid)) * 3.0


def draw_classification_errors(ax, d, fit_n_max: int = 884):
    n = d["n"]; fit_mask = n <= fit_n_max
    n_fit = n[fit_mask]
    xd = np.geomspace(n_fit[0], n_fit[-1], 100)
    log_dxd = np.log(xd / n_fit[0])
    taper = 0.5 + 0.5 * np.abs(log_dxd) / max(log_dxd[-1], 1.0)
    y_min, y_max = np.inf, -np.inf
    for color, lab, arr in [
        (C_RMSE,   "RMSE",          d["rmse"]),
        (C_POOL,   "Pooling",       d["pool"]),
        (C_LOGIT,  "Logit",         d["logit"]),
        (C_DISAGR, "Disagreement",  1.0 - d["acc_vs_dense"]),
    ]:
        y_norm, slope, band = _fit_pinned(arr, fit_mask, n)
        beta = -slope
        y_line = np.exp(slope * log_dxd)
        ax.plot(xd, y_line, "-", color=color, lw=2.4, zorder=2)
        ax.fill_between(xd, y_line * np.exp(-band * taper), y_line * np.exp(band * taper),
                        color=color, alpha=0.10, linewidth=0, zorder=1)
        ax.scatter(n_fit, y_norm, color=color, s=35, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=5,
                   label=f"{lab} (β={beta:.2f})")
        y_min = min(y_min, float(y_norm.min()))
        y_max = max(y_max, float(y_norm.max()))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(n_fit[0], n_fit[-1])
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("n (keys per query)")
    ax.set_ylabel("Error rates")
    ax.legend(loc="lower left", framealpha=0.9, handletextpad=0.35,
              labelspacing=0.45, borderpad=0.45)


def draw_accuracy_agreement(ax, d):
    n = d["n"]
    ax.plot(n, d["acc_labels"], "o-", color=C_ACC, ms=6,
            markeredgecolor="white", markeredgewidth=0.5, lw=2.0, label="Accuracy")
    ax.plot(n, d["acc_vs_dense"], "s-", color=C_AGR, ms=6,
            markeredgecolor="white", markeredgewidth=0.5, lw=2.0, label="Agreement rate")
    ax.axhline(d["acc_dense"], ls="--", color="gray", lw=1.4,
               label="Reference accuracy")
    ax.set_xscale("log")
    ax.set_xlim(n.min(), n.max())
    y_min = float(min(d["acc_labels"].min(), d["acc_vs_dense"].min()))
    y_max = float(max(d["acc_labels"].max(), d["acc_vs_dense"].max()))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("n (keys per query)")
    ax.legend(loc="lower right", framealpha=0.9, handletextpad=0.6, labelspacing=0.6)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", type=Path,
                   default=Path("results/downstream_global/metrics.json"))
    p.add_argument("--out-dir", type=Path, default=Path("figures/downstream_global"))
    p.add_argument("--fit-n-max", type=int, default=884)
    p.add_argument("--figsize", nargs=2, type=float, default=(3.5, 6.0),
                   help="(width, height) in inches. Default is the camera-ready icml_3p5x6 layout.")
    args = p.parse_args()

    apply_paper_style()
    d = _load(args.metrics)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    draw_classification_errors(ax, d, fit_n_max=args.fit_n_max)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "classification_errors")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    draw_accuracy_agreement(ax, d)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "accuracy_agreement")
    plt.close(fig)
    print(f"saved {args.out_dir}/classification_errors.{{png,pdf}} "
          f"and {args.out_dir}/accuracy_agreement.{{png,pdf}}")


if __name__ == "__main__":
    main()
