#!/usr/bin/env python3
"""
Plot end-to-end error vs n with one curve per H/Lambda_A cluster, fit beta per
cluster, run Spearman tests, and rank candidate aggregations by separation score.

Inputs (loaded by concatenating shards):
  results_horizon/H_agg{shard}.npy    — (n_ex, num_layers)         two-sided H
  results_horizon/H_heads{shard}.npy  — (n_ex, num_layers, num_heads)
  results_horizon/Lambda_A_agg{shard}.npy — (n_ex, num_layers)
  results_horizon/Lambda_A_heads{shard}.npy
  results_window_random/err_pool/err_logit/err_kl/sparse_preds/dense_logits/labels — (n_ex, nb_n, ...)
  results_window_random/n_values.npy

Output (one figure per (error metric, family) — winners only, NOT per candidate):
  <results_dir>/plots/<metric>__winner.png       (overall winner across families)
  <results_dir>/plots/<metric>__H_winner.png  (best H candidate for this metric)
  <results_dir>/plots/<metric>__LA_winner.png    (best LA candidate for this metric)
  <results_dir>/plots/separation_summary.csv     (full ranked table, all candidates)
  <results_dir>/plots/spearman_per_n.csv         (rho_s(n) per candidate, metric)
  <results_dir>/plots/winner.json                (overall winner per metric)
  <results_dir>/plots/family_winners.json        (best per family per metric)

Usage:
    python plot_window_H_clusters.py \
        --window-dir results_window_random \
        --horizon-dir results_horizon \
        --out-dir results_window_random
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr

# Red → orange → yellow gradient matching the BERT convergence-stars plots
# (low H = red, mid = orange, high = yellow).
C_LOW, C_MID, C_HIGH = "#B81D24", "#D97030", "#FFAB20"

# Fit β only on the linear (power-law) regime — same convention as
# 01_iid_sparse/plot.py:FIT_N_MAX. 
FIT_N_MAX_DEFAULT = 884


def concat_shards(directory, prefix):
    """Concatenate {prefix}_shard_*.npy in start-idx order, or load {prefix}.npy if it exists."""
    direct = os.path.join(directory, f"{prefix}.npy")
    if os.path.isfile(direct):
        return np.load(direct)
    files = []
    for f in os.listdir(directory):
        if f.startswith(f"{prefix}_shard_") and f.endswith(".npy"):
            tail = f[len(f"{prefix}_shard_"):-len(".npy")]
            try:
                start = int(tail.split("_")[0])
            except ValueError:
                continue
            files.append((start, f))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No shards or direct file for {prefix} in {directory}")
    parts = [np.load(os.path.join(directory, f)) for _, f in files]
    return np.concatenate(parts, axis=0)


def fit_beta(n_arr, errors):
    ok = (errors > 1e-15) & np.isfinite(errors)
    if ok.sum() < 3:
        return None, None
    s, _, r, _, _ = linregress(np.log(n_arr[ok].astype(float)), np.log(errors[ok]))
    return -s, r ** 2


def tertile_split(score):
    q33, q66 = np.quantile(score, [1/3, 2/3])
    low  = score <= q33
    high = score >= q66
    mid  = ~(low | high)
    return low, mid, high


def _family_aggregations(prefix, agg, heads):
    """Mirror the H_* candidate set for a generic per-(example, layer) array.

    agg: (n_ex, L), heads: (n_ex, L, num_heads). Returns dict of named scores.
    """
    out = {}
    L = agg.shape[1]
    out[f"{prefix}_mean_all"]   = agg.mean(axis=1)
    out[f"{prefix}_max_all"]    = agg.max(axis=1)
    out[f"{prefix}_mean_deep"]  = agg[:, L//2:].mean(axis=1)
    out[f"{prefix}_logsum_all"] = np.log(np.maximum(agg, 1e-30)).sum(axis=1)
    for l in range(L//2, L):
        out[f"{prefix}_layer_{l}"] = agg[:, l]
    out[f"{prefix}_heads_max"]  = heads.max(axis=(1, 2))
    return out


def candidate_aggregations(H_agg, H_heads, La_agg, La_heads):
    """Return dict candidate_name -> per-example score (n_ex,).

    Two families:
      H_* — two-sided horizon  ||L^T A L||_2 = ||Σ^{1/2} A Σ^{1/2}||_2
      LA_*   — Lambda_A = lambda_max(P_A Σ P_A)
    """
    cands = {}
    cands.update(_family_aggregations("H", H_agg, H_heads))
    cands.update(_family_aggregations("LA",   La_agg,   La_heads))
    return cands


def cluster_curves(err, low, mid, high):
    """err: (n_ex, nb_n).  Returns (3, nb_n) average curves."""
    return np.stack([err[low].mean(axis=0),
                     err[mid].mean(axis=0),
                     err[high].mean(axis=0)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-dir", default="results_window_random")
    ap.add_argument("--horizon-dir", default="results_horizon")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-fit-max", type=int, default=FIT_N_MAX_DEFAULT,
                    help=f"Fit beta on n <= this. Default = {FIT_N_MAX_DEFAULT} "
                         "(linear-regime cutoff, matches 01_iid_sparse).")
    ap.add_argument("--spearman-thresh", type=float, default=0.3)
    ap.add_argument("--p-thresh", type=float, default=0.01)
    ap.add_argument("--gap-thresh", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = args.out_dir or args.window_dir
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("[Load] Horizon data")
    H_agg   = concat_shards(args.horizon_dir, "H_agg")
    H_heads = concat_shards(args.horizon_dir, "H_heads")
    La_agg     = concat_shards(args.horizon_dir, "Lambda_A_agg")
    La_heads   = concat_shards(args.horizon_dir, "Lambda_A_heads")
    print(f"  H_agg {H_agg.shape}  Lambda_A_agg {La_agg.shape}")

    print("[Load] Window+random data")
    n_values  = np.load(os.path.join(args.window_dir, "n_values.npy"))
    err_pool  = concat_shards(args.window_dir, "err_pool")
    err_logit = concat_shards(args.window_dir, "err_logit")
    preds     = concat_shards(args.window_dir, "sparse_preds")
    dense_l   = concat_shards(args.window_dir, "dense_logits")
    labels    = concat_shards(args.window_dir, "labels")
    print(f"  err_pool {err_pool.shape}  preds {preds.shape}  dense_logits {dense_l.shape}")

    n_ex = err_pool.shape[0]
    assert H_agg.shape[0] >= n_ex, "Need at least n_ex H entries"
    H_agg, H_heads = H_agg[:n_ex], H_heads[:n_ex]
    La_agg,   La_heads   = La_agg[:n_ex],   La_heads[:n_ex]

    dense_pred = dense_l.argmax(axis=1)
    err_disagree = (preds != dense_pred[:, None, None]).mean(axis=2).astype(float)

    # RMSE per class element of the logit prediction = ||Δ logit||_2 / sqrt(K).
    # Same β as logit_l2 by construction; reports the typical per-class deviation
    # in interpretable units. K = num classes (11 for arxiv-classification).
    K = dense_l.shape[1]
    rmse_logit = err_logit / np.sqrt(K)

    metrics = {
        "pool_l2":      err_pool,
        "logit_l2":     err_logit,
        "rmse":         rmse_logit,
        "disagreement": err_disagree,
    }

    # Inclusive <= so the cutoff itself is included (matches 01_iid_sparse).
    fit_mask = n_values <= args.n_fit_max
    n_fit = n_values[fit_mask]
    print(f"[Fit] β fit on {len(n_fit)} points n in {n_fit.tolist()}")

    cands = candidate_aggregations(H_agg, H_heads, La_agg, La_heads)
    print(f"[Cands] {len(cands)} candidates: {list(cands.keys())}")

    def _family(c):
        return "H" if c.startswith("H_") else ("LA" if c.startswith("LA_") else "other")

    def _save_panel(mname, cname, row, curves, betas, fname):
        """One panel per (metric, candidate). Pinned no-intercept fit on the
        fit range (n ≤ n_fit_max). Each cluster curve is normalized to 1.0 at
        n_min so all three start at the same point on the y-axis. Shaded band
        is ±σ_slope · |log(n/n_min)| — shrinks to zero at the first point and
        widens with log-distance.
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        n_fit_arr = n_fit.astype(float)
        log_dx = np.log(n_fit_arr) - np.log(n_fit_arr[0])
        S_xx   = float(np.sum(log_dx ** 2))
        for lab, col in [("Low", C_LOW), ("Mid", C_MID), ("High", C_HIGH)]:
            k = {"Low": 0, "Mid": 1, "High": 2}[lab]
            y_full = curves[k, fit_mask]
            y_norm = y_full / y_full[0]
            log_y  = np.log(y_norm)
            slope  = float(np.sum(log_dx * log_y) / S_xx)
            beta   = -slope
            # Constant-width band in log-y: mean over n of relative SE × 3.
            cluster_idx = {"Low": "low", "Mid": "mid", "High": "high"}  # only for symmetry
            err_cluster = curves[k][fit_mask] / curves[k][fit_mask][0]   # already a mean
            # Use within-curve residual scatter as a proxy for the band width.
            resid = log_y - slope * log_dx
            avg_log_std = float(np.mean(np.abs(resid))) * 3.0

            xd = np.geomspace(n_fit_arr[0], n_fit_arr[-1], 100)
            y_line = (xd / n_fit_arr[0]) ** slope
            log_dxd = np.log(xd) - np.log(n_fit_arr[0])
            log_dxd_max = log_dxd[-1] if log_dxd[-1] > 0 else 1.0
            band = avg_log_std * (0.5 + 0.5 * np.abs(log_dxd) / log_dxd_max)
            ax.plot(xd, y_line, "-", color=col, linewidth=2.4, zorder=2)
            ax.fill_between(xd, y_line * np.exp(-band),
                            y_line * np.exp(band),
                            color=col, alpha=0.10, linewidth=0, zorder=1)
            ax.scatter(n_fit_arr, y_norm, color=col, s=35, alpha=0.85,
                       edgecolors="white", linewidths=0.5, zorder=5,
                       label=f"{lab}  β={beta:.3f}")
        ax.set_xlim(n_fit_arr[0], n_fit_arr[-1])
        ax.set_xlabel("n (keys per query)", fontsize=16)
        ax.set_ylabel(mname, fontsize=16)
        ax.set_title(f"{mname} — {cname}\n"
                     f"range β={row['range_beta']:.3f}  "
                     f"ρ̄={row['mean_rho']:.3f}  "
                     f"{'PASS' if row['passes_gate'] else 'fail'}",
                     fontsize=13)
        ax.legend(fontsize=11, framealpha=0.9, loc="lower left")
        ax.tick_params(labelsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, fname), dpi=110)
        plt.close(fig)

    summary_rows = []
    spearman_rows = []
    winners = {}              # overall best per metric (any family)
    family_winners = {        # best within each family per metric
        "H": {}, "LA": {},
    }

    for mname, err in metrics.items():
        best_overall = None         # (score_tuple, cname, row, curves, betas)
        best_per_family = {"H": None, "LA": None}

        for cname, score in cands.items():
            low, mid, high = tertile_split(score)
            curves = cluster_curves(err, low, mid, high)

            betas, r2s = [], []
            for k in range(3):
                b, r2 = fit_beta(n_fit, curves[k, fit_mask])
                betas.append(b); r2s.append(r2)
            b_low, b_mid, b_high = betas
            r2_mean = float(np.mean([r for r in r2s if r is not None])) \
                      if all(r is not None for r in r2s) else None

            rhos, ps = [], []
            for j in range(len(n_values)):
                if fit_mask[j]:
                    finite = np.isfinite(err[:, j]) & np.isfinite(score)
                    if finite.sum() < 10:
                        continue
                    rho, p = spearmanr(score[finite], err[finite, j])
                    rhos.append(rho); ps.append(p)
                    spearman_rows.append({
                        "metric": mname, "candidate": cname,
                        "n": int(n_values[j]), "rho": float(rho), "p": float(p),
                    })
            mean_rho = float(np.mean(rhos)) if rhos else None
            max_p    = float(np.max(ps))    if ps else None

            if None in (b_low, b_mid, b_high, mean_rho, r2_mean):
                continue
            range_b = b_low - b_high
            min_gap = min(b_mid - b_high, b_low - b_mid)
            monotone = (b_high < b_mid < b_low)
            spearman_ok = (mean_rho >= args.spearman_thresh and
                           max_p <= args.p_thresh)
            passes = monotone and (min_gap >= args.gap_thresh) and spearman_ok

            row = {
                "metric": mname, "candidate": cname,
                "beta_low": b_low, "beta_mid": b_mid, "beta_high": b_high,
                "range_beta": range_b, "min_gap": min_gap,
                "r2_mean": r2_mean,
                "mean_rho": mean_rho, "max_p": max_p,
                "passes_gate": passes,
            }
            summary_rows.append(row)

            score_tuple = (passes, range_b, mean_rho)
            if best_overall is None or score_tuple > best_overall[0]:
                best_overall = (score_tuple, cname, row, curves, (b_low, b_mid, b_high))
            fam = _family(cname)
            if fam in best_per_family:
                cur = best_per_family[fam]
                if cur is None or score_tuple > cur[0]:
                    best_per_family[fam] = (score_tuple, cname, row, curves, (b_low, b_mid, b_high))

        # Overall winner panel
        if best_overall is not None:
            _, cname, row, curves, betas = best_overall
            winners[mname] = {"candidate": cname, **row}
            _save_panel(mname, cname, row, curves, betas, f"{mname}__winner.png")

        # Per-family winner panels (H, LA). One PNG per (metric, family).
        for fam, entry in best_per_family.items():
            if entry is None:
                continue
            _, cname, row, curves, betas = entry
            family_winners[fam][mname] = {"candidate": cname, **row}
            _save_panel(mname, cname, row, curves, betas, f"{mname}__{fam}_winner.png")

    import csv
    with open(os.path.join(plot_dir, "separation_summary.csv"), "w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in sorted(summary_rows,
                            key=lambda r: (-int(r["passes_gate"]),
                                           -r["range_beta"])):
                w.writerow(r)

    with open(os.path.join(plot_dir, "spearman_per_n.csv"), "w", newline="") as f:
        if spearman_rows:
            w = csv.DictWriter(f, fieldnames=list(spearman_rows[0].keys()))
            w.writeheader()
            w.writerows(spearman_rows)

    with open(os.path.join(plot_dir, "winner.json"), "w") as f:
        json.dump(winners, f, indent=2, default=float)

    with open(os.path.join(plot_dir, "family_winners.json"), "w") as f:
        json.dump(family_winners, f, indent=2, default=float)

    print("\n=== Overall winner (per metric, any family) ===")
    for m, w in winners.items():
        print(f"  {m:14s}  {w['candidate']:20s}  "
              f"range β={w['range_beta']:.3f}  ρ̄={w['mean_rho']:.3f}  "
              f"{'PASS' if w['passes_gate'] else 'fail'}")
    for fam, by_m in family_winners.items():
        print(f"\n=== Best {fam} candidate (per metric) ===")
        for m, w in by_m.items():
            print(f"  {m:14s}  {w['candidate']:20s}  "
                  f"range β={w['range_beta']:.3f}  ρ̄={w['mean_rho']:.3f}  "
                  f"{'PASS' if w['passes_gate'] else 'fail'}")
    print(f"\nPlots and CSVs in {plot_dir}")


if __name__ == "__main__":
    main()
