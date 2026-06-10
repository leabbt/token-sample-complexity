"""Uniform-on-sphere Monte-Carlo convergence experiment in dimension d=50.

For each scale in `--scales`, build random Q, K, V projections at that scale,
compute the (closed-form) reference covariance, then sweep over the i.i.d.
sample size `n` and fit the convergence rate of both mean and covariance errors.

Writes one CSV per scale plus a `_meta.csv` per scale containing the horizon
and fitted slopes, so the two plot scripts can collect every scale together.

Example
-------
    python experiments/uniform_sphere_d50/run.py --quick
    python experiments/uniform_sphere_d50/run.py --d 50 --k 200 \\
        --n_min 10000 --n_max 100000 --nb-points 12 \\
        --scale-min 1e1 --scale-max 1e5 --num-scales 12
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from tsc.sphere import SphereConfig, run_sphere_experiment


def _save_pair(df, meta, scale: float, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"sphere_d{meta['d']}_scale{scale:.4e}_H{meta['horizon']:.4e}"
    df.to_csv(output_dir / f"{name}.csv", index=False)
    import pandas as pd
    pd.DataFrame([meta]).to_csv(output_dir / f"{name}_meta.csv", index=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, default=50)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--n_min", type=int, default=10_000)
    p.add_argument("--n_max", type=int, default=100_000)
    p.add_argument("--nb-points", type=int, default=12)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--scale-min", type=float, default=1e1)
    p.add_argument("--scale-max", type=float, default=1e5)
    p.add_argument("--num-scales", type=int, default=5)
    p.add_argument("--mode", default="both", choices=["mean", "cov", "both"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("results/uniform_sphere_d50"))
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.d = 50
        args.n_min, args.n_max, args.nb_points, args.k = 500, 4000, 4, 5
        args.num_scales = 3
        args.scale_min, args.scale_max = 1.0, 100.0

    scales = np.geomspace(args.scale_min, args.scale_max, args.num_scales)
    for scale in scales:
        # Equal Q and K scales (canonical choice for the d=50 figures).
        cfg = SphereConfig(
            d=args.d, n_min=args.n_min, n_max=args.n_max,
            nb_points=args.nb_points, k=args.k,
            W_q_scale=float(scale), W_k_scale=float(scale), W_v_scale=1.0,
            dist="sphere", radius=args.radius, mode=args.mode, seed=args.seed,
        )
        df, meta = run_sphere_experiment(cfg)
        _save_pair(df, meta, float(scale), args.output_dir)
        print(f"scale={scale:.3e}  H={meta['horizon']:.4e}  "
              f"β_m={meta.get('mean_slope', float('nan')):.3f}  "
              f"β_c={meta.get('cov_slope', float('nan')):.3f}")


if __name__ == "__main__":
    main()
