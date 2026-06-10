"""
Shared loader for horizon-vs-slope plots.

Reads one or more `horizons_<dataset>.csv` files (one row per (dataset, layer),
written by experiments/horizons/compute_horizons.py) and joins them with the
slope info extracted from the per-layer `hyperparameters.csv` files produced by
the IID convergence experiment.

Returned DataFrame columns (one row per surviving (dataset, layer)):
    dataset, layer_id, H, max_eigen, k,
    slope_mean, slope_mean_se, slope_mean_se_corrected,
    slope_cov,  slope_cov_se,  slope_cov_se_corrected,

`H` is whichever horizon column was requested (default 'Horizon_new').

Horizon definitions (Cholesky-based, with L the lower-triangular factor of Σ):
    Horizon_old = ||L^T A||_2       ( = ||Σ^{1/2} A||_2 )
    Horizon_new = ||L^T A L||_2     ( = ||Σ^{1/2} A Σ^{1/2}||_2 )
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


HORIZON_COLS = ("Horizon_old", "Horizon_new")


def _read_horizons(horizons_paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for p in horizons_paths:
        p = Path(p)
        if not p.exists():
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(
            f"No horizons CSVs found among: {[str(p) for p in horizons_paths]}"
        )
    return pd.concat(frames, ignore_index=True)


def _slope_per_layer(
    search_dirs: Sequence[Path],
    pattern: str,
    exclude_layers: Iterable[int] = (),
) -> Dict[int, dict]:
    """Walk search_dirs for hyperparameters.csv files matching `pattern`
    and keep the highest-k entry per layer."""
    exclude_layers = set(exclude_layers)
    by_layer: Dict[int, dict] = {}
    for search_dir in search_dirs:
        search_dir = Path(search_dir)
        if not search_dir.exists():
            continue
        for hp_file in search_dir.rglob("hyperparameters.csv"):
            if pattern not in str(hp_file):
                continue
            try:
                df = pd.read_csv(hp_file)
                if df.empty:
                    continue
                row = df.iloc[0]
                layer_id = int(row["layer_id"])
                if layer_id in exclude_layers:
                    continue
                k = int(row["k"])
                cur = by_layer.get(layer_id)
                if cur is None or k > cur["k"]:
                    by_layer[layer_id] = {
                        "layer_id": layer_id,
                        "k": k,
                        "max_eigen": row.get("max_eigen", np.nan),
                        "slope_mean": row["slope_mean"],
                        "slope_mean_se": row["slope_mean_se"],
                        "slope_cov": row["slope_cov"],
                        "slope_cov_se": row["slope_cov_se"],
                    }
            except Exception:
                continue
    return by_layer


def load_horizons_and_slopes(
    horizons_paths: Sequence[Path],
    search_dirs_by_dataset: Dict[str, Sequence[Path]],
    pattern_by_dataset: Dict[str, str],
    exclude_layers_by_dataset: Optional[Dict[str, Iterable[int]]] = None,
    horizon_col: str = "Horizon_new",
) -> pd.DataFrame:
    """Join horizons.csv rows with the highest-k slope info per (dataset, layer).

    Args:
        horizons_paths: list of horizons_<dataset>.csv paths to concat.
        search_dirs_by_dataset: { dataset_name: [Path, ...] } — where to look
            for hyperparameters.csv files. Same shape as the existing
            SEARCH_DIRS dict in plot scripts.
        pattern_by_dataset: { dataset_name: substring } — used to filter
            hyperparameters.csv files (matches against the file's full path).
        exclude_layers_by_dataset: optional { dataset_name: iterable[layer_id] }
            of layers to drop from the final DataFrame.
        horizon_col: which column from horizons.csv to expose as `H`.
            One of {'Horizon_old', 'Horizon_new'}.

    Returns:
        DataFrame sorted by (dataset, layer_id) with columns described in the
        module docstring.
    """
    if horizon_col not in HORIZON_COLS:
        raise ValueError(f"horizon_col must be one of {HORIZON_COLS}, got {horizon_col!r}")

    horizons = _read_horizons(horizons_paths)
    if horizon_col not in horizons.columns:
        raise KeyError(
            f"column {horizon_col!r} missing from horizons CSV. "
            f"Available: {list(horizons.columns)}"
        )

    exclude = exclude_layers_by_dataset or {}
    rows: List[dict] = []
    for dataset, dirs in search_dirs_by_dataset.items():
        pattern = pattern_by_dataset.get(dataset, dataset)
        slopes = _slope_per_layer(dirs, pattern, exclude.get(dataset, ()))

        ds_horizons = horizons[horizons["dataset"] == dataset].set_index("layer")
        for layer_id, slope_info in slopes.items():
            if layer_id not in ds_horizons.index:
                # No horizon row for this (dataset, layer) — silently skip.
                continue
            h_row = ds_horizons.loc[layer_id]
            k = slope_info["k"]
            rows.append({
                "dataset": dataset,
                "layer_id": layer_id,
                "H": float(h_row[horizon_col]),
                "max_eigen": float(h_row.get("max_eigen", slope_info["max_eigen"])),
                "k": k,
                "slope_mean": slope_info["slope_mean"],
                "slope_mean_se": slope_info["slope_mean_se"],
                "slope_mean_se_corrected": slope_info["slope_mean_se"] / np.sqrt(k),
                "slope_cov": slope_info["slope_cov"],
                "slope_cov_se": slope_info["slope_cov_se"],
                "slope_cov_se_corrected": slope_info["slope_cov_se"] / np.sqrt(k),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "dataset", "layer_id", "H", "max_eigen", "k",
            "slope_mean", "slope_mean_se", "slope_mean_se_corrected",
            "slope_cov",  "slope_cov_se",  "slope_cov_se_corrected",
        ])

    return pd.DataFrame(rows).sort_values(["dataset", "layer_id"]).reset_index(drop=True)


def default_horizons_paths(repo_root: Path, datasets: Sequence[str]) -> List[Path]:
    """Standard place to find horizons CSVs. compute_horizons.py writes one
    row per (dataset, layer) into `horizons_<dataset>_layer<L>.csv`; this glob
    picks up every such file for each requested dataset.

    Returns an empty list if none are present (the caller will then raise a
    clear error). Files are concatenated by `_read_horizons`.
    """
    base = Path(repo_root) / "results" / "horizons"
    paths: List[Path] = []
    for d in datasets:
        paths.extend(sorted(base.glob(f"horizons_{d}_layer*.csv")))
    return paths
