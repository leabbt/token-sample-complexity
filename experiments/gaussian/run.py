"""Synthetic Gaussian-tokens convergence experiment.

Sweeps an attention-scale factor that controls the horizon
`H = ‖Σ^{1/2} A Σ^{1/2}‖₂`. At each horizon we run a Monte-Carlo subsample sweep,
fit the convergence rate of the mean and covariance errors, and save one row
per horizon to a CSV.

Example
-------
    python experiments/gaussian/run.py --quick
    python experiments/gaussian/run.py --d 50 --rho 0.1 --k 200 \\
        --n_min 1000 --n_max 30000 --nb_tot 12 --num-scales 12 \\
        --scale-min 0.05 --scale-max 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tsc.gaussian import GaussianConfig, run_gaussian_horizon_sweep


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, default=50, help="embedding dimension")
    p.add_argument("--rho", type=float, default=0.1, help="Σ = ρ·I_d")
    p.add_argument("--n-reference", type=int, default=100_000,
                   help="tokens used to build the finite reference distribution")
    p.add_argument("--n_min", type=int, default=1000)
    p.add_argument("--n_max", type=int, default=30_000)
    p.add_argument("--nb_tot", type=int, default=12)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--num-scales", type=int, default=12)
    p.add_argument("--scale-min", type=float, default=0.05)
    p.add_argument("--scale-max", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results/gaussian/results.csv"))
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.d = 8
        args.n_reference = 2_000
        args.n_min, args.n_max, args.nb_tot, args.k = 200, 1500, 4, 5
        args.num_scales = 4

    scales = np.geomspace(args.scale_min, args.scale_max, args.num_scales)
    cfg = GaussianConfig(
        d=args.d, rho=args.rho, n_reference=args.n_reference,
        n_min=args.n_min, n_max=args.n_max, nb_tot=args.nb_tot, k=args.k,
        scales=scales, seed=args.seed,
    )
    df = run_gaussian_horizon_sweep(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
