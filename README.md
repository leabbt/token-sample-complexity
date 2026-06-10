# Code to reproduce the ICML 2026 Paper: "_Token Sample Complexity of Attention_"

## Requirements

Python 3.10 or newer.

## Install

From the repo root:

```bash
pip install -e .
```

This installs the `tsc` shared library (the experiment scripts import it as
`from tsc import ...`) and its runtime dependencies (`torch`, `transformers`,
`datasets`, `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `pyyaml`,
`scikit-learn`). The first run downloads BigBird / BERT from HuggingFace into
the local HF cache.

## Reproducing the experiments/figures of the paper

Each runner accepts `--quick` for a tiny local test. Paper parameters are
shown below. Plot scripts write both `.png` (300 dpi) and `.pdf` (vector).

### BigBird, i.i.d. token subsampling — convergence curves and slope vs horizon

```bash
PYTHONPATH=src python experiments/bigbird_iid/run.py \
    --source wiki --language en --layer 0 \
    --N 122 --n_min 1000 --n_max 3000 --nb_tot 12 --k 500 \
    --output-dir results/bigbird_iid/wiki_en_k500

PYTHONPATH=src python experiments/bigbird_iid/plot_convergence.py \
    --results-dir results/bigbird_iid/wiki_en_k500
PYTHONPATH=src python experiments/bigbird_iid/plot_slopes_vs_H.py \
    --results-dir results/bigbird_iid/wiki_en_k500
```

Plots are in the folder `figures/`.

### BigBird, window + r random tokens — convergence curves and slope vs horizon

```bash
PYTHONPATH=src python experiments/bigbird_window/run.py \
    --source wiki --language en --layer 0 \
    --N 122 --w_min 500 --w_max 1500 --nb_tot 12 --k 500 --random-tokens 50 \
    --output-dir results/bigbird_window/wiki_en_k500

PYTHONPATH=src python experiments/bigbird_window/plot_convergence.py \
    --results-dir results/bigbird_window/wiki_en_k500
PYTHONPATH=src python experiments/bigbird_window/plot_slopes_vs_H.py \
    --results-dir results/bigbird_window/wiki_en_k500
```

### BERT, i.i.d. token subsampling — convergence curves and slope vs horizon

```bash
PYTHONPATH=src python experiments/bert/run.py \
    --source wiki --language en --layer 0 \
    --max-length 512 --n_min 64 --n_max 384 --nb_tot 10 --k 500 \
    --output-dir results/bert/wiki_en_k500

PYTHONPATH=src python experiments/bert/plot_convergence.py \
    --results-dir results/bert/wiki_en_k500
PYTHONPATH=src python experiments/bert/plot_slopes_vs_H.py \
    --results-dir results/bert/wiki_en_k500
```

### Synthetic Gaussian tokens

```bash
PYTHONPATH=src python experiments/gaussian/run.py \
    --d 50 --rho 0.1 --n-reference 100000 \
    --n_min 1000 --n_max 30000 --nb_tot 12 --k 200 \
    --num-scales 12 --scale-min 0.05 --scale-max 50 \
    --output results/gaussian/results.csv

PYTHONPATH=src python experiments/gaussian/plot.py \
    --results results/gaussian/results.csv
```

### Uniform distribution on the sphere, dimension 50

```bash
PYTHONPATH=src python experiments/uniform_sphere_d50/run.py \
    --d 50 --k 200 \
    --n_min 10000 --n_max 100000 --nb-points 12 \
    --num-scales 12 --scale-min 1e1 --scale-max 1e5 --mode both \
    --output-dir results/uniform_sphere_d50

PYTHONPATH=src python experiments/uniform_sphere_d50/plot_convergence.py \
    --results-dir results/uniform_sphere_d50
PYTHONPATH=src python experiments/uniform_sphere_d50/plot_slopes_vs_H.py \
    --results-dir results/uniform_sphere_d50
```

### Downstream classification — global

Errors, accuracy, and agreement rate vs sample size on `ccdv/arxiv-classification`.
Requires a fine-tuned BigBird classifier; pass it via `--model`.

```bash
PYTHONPATH=src python experiments/downstream_global/run.py \
    --model <path-to-finetuned-bigbird> \
    --n-examples 500 --k-mc 20 \
    --n-min 64 --n-max 4096 --nb-points 20 \
    --results-dir results/downstream_global

PYTHONPATH=src python experiments/downstream_global/plot.py \
    --metrics results/downstream_global/metrics.json \
    --out-dir figures/downstream_global
```

### Downstream classification — per horizon

Two stages. First compute the per-example horizon, then run the per-sample
sweep, then stratify by horizon tertiles.

```bash
PYTHONPATH=src python experiments/downstream_per_horizon/compute_horizons.py \
    --model <path-to-finetuned-bigbird> \
    --results-dir results/downstream_per_horizon/horizon

PYTHONPATH=src python experiments/downstream_per_horizon/run_persample.py \
    --model <path-to-finetuned-bigbird> \
    --results-dir results/downstream_per_horizon/persample

PYTHONPATH=src python experiments/downstream_per_horizon/plot.py \
    --window-dir results/downstream_per_horizon/persample \
    --horizon-dir results/downstream_per_horizon/horizon \
    --out-dir results/downstream_per_horizon/persample
```

### t-SNE figure

Open the notebook and run all cells:

```bash
jupyter notebook notebooks/tsne_figure.ipynb
```

The figure is written to `figures/tsne_subsample_convergence.{png,pdf}`. Set
`SMOKE_TEST = True` at the top of the notebook for a small local run.

## Horizon computation

The horizon is computed as

```
L = cholesky(Σ + 1e-6·I)        # L Lᵀ = Σ + jitter (lower-triangular)
H = ‖Lᵀ A L‖₂                    # spectral norm
```

with `A = Wkᵀ Wq`, and `Σ` either the empirical covariance of the layer-input
tokens (BigBird/BERT/downstream) or the analytic covariance of the synthetic
distribution (Gaussian/sphere).

## Layout

```
src/tsc/                       library (attention, sampling, fit, horizons, ...)
experiments/<name>/            one folder per experiment, runner + plot scripts
notebooks/                     t-SNE figure
```

Cite
----

If you use this code in your project, please cite:

Bohbot, L., Letrouit, C., Peyré, G., & Vialard, F.-X.
Token Sample Complexity of Attention.
*Proceedings of the 43rd International Conference on Machine Learning*,
Seoul, South Korea. PMLR 306, 2026.
https://arxiv.org/abs/2512.10656

```bibtex
@inproceedings{bohbot2026token,
  title     = {Token Sample Complexity of Attention},
  author    = {Bohbot, L{\'e}a and Letrouit, Cyril and Peyr{\'e}, Gabriel and Vialard, Fran{\c{c}}ois-Xavier},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {306},
  year      = {2026},
  address   = {Seoul, South Korea},
  url       = {https://arxiv.org/abs/2512.10656},
}
```
