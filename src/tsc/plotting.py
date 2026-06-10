"""Shared plot styling for paper figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colors used consistently across the paper figures.
MEAN_COLOR = "#7E1FA2"       # purple — mean-error series
COV_COLOR = "#E67E22"        # orange — covariance-error series
THEORY_COLOR = "darkred"
REFERENCE_GRAY = "0.55"

# 12-layer color cycle (BigBird/BERT layers 0-11).
LAYER_CMAP = plt.get_cmap("viridis", 12)


def apply_paper_style() -> None:
    """Apply the matplotlib rcParams used across paper figures."""
    mpl.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "lines.linewidth": 1.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(fig, out_path) -> None:
    """Save both PNG (300 dpi) and PDF (vector) under the same stem."""
    from pathlib import Path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
