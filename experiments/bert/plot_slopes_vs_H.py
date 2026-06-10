"""Slope-vs-horizon plot: fitted convergence rates `|β_mean|` and `|β_cov|`
as a function of the per-layer horizon H. Every layer found
under `--results-dir` is plotted; no layers are dropped, no points are starred."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tsc.plotting import apply_paper_style, save_figure, MEAN_COLOR, COV_COLOR, THEORY_COLOR


def _collect(results_dir: Path):
    """Aggregate (horizon, slope_m, slope_c) over every meta.json."""
    horizons, slopes_m, slopes_c, se_m, se_c, labels = [], [], [], [], [], []
    for meta_path in results_dir.glob("**/meta.json"):
        with open(meta_path) as f:
            meta = json.load(f)
        horizons.append(meta["horizon"])
        slopes_m.append(meta["slope_m"])
        slopes_c.append(meta["slope_c"])
        se_m.append(meta.get("slope_m_se", 0.0))
        se_c.append(meta.get("slope_c_se", 0.0))
        labels.append(f"{meta['source']}/{meta['language']} L{meta['layer_id']}")
    return (np.array(horizons), np.array(slopes_m), np.array(slopes_c),
            np.array(se_m), np.array(se_c), labels)


def plot(results_dir: Path, out_path: Path, theory: bool = True):
    apply_paper_style()
    H, sm, sc, em, ec, _ = _collect(results_dir)
    if H.size == 0:
        raise SystemExit(f"No meta.json under {results_dir}")

    fig, (ax_m, ax_c) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax_m.errorbar(H, np.abs(sm), yerr=em, fmt="o", color=MEAN_COLOR, ms=6,
                  label=r"$|\beta_{\mathrm{mean}}|$")
    ax_c.errorbar(H, np.abs(sc), yerr=ec, fmt="s", color=COV_COLOR, ms=6,
                  label=r"$|\beta_{\mathrm{cov}}|$")

    if theory:
        x = np.geomspace(max(H.min(), 1e-3), max(H.max(), 1.0) * 2, 500)
        y_th = 1.0 / (2.0 * (1.0 + x**2))
        for ax in (ax_m, ax_c):
            ax.plot(x, y_th, color=THEORY_COLOR, lw=1.8,
                    label=r"$\frac{1}{2(1+H^2)}$")

    for ax, title in [(ax_m, "Mean error"), (ax_c, "Covariance error")]:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axhline(0.5, color="0.6", ls="--", lw=1, alpha=0.8)
        ax.set_xlabel(r"Horizon $H$")
        ax.set_ylabel(r"$|\beta|$")
        ax.set_title(title)
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"saved {out_path.with_suffix('.png')}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--out", type=Path,
                   default=Path("figures/bert_slopes_vs_H"))
    p.add_argument("--no-theory", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.results_dir, args.out, theory=not args.no_theory)
