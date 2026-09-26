# DSS-GNN

Code for **A Unified Uncertainty Representation for Graph Neural Networks via
Doubly-Spectral Stochastic Expansion** (NeurIPS 2026) by Fred Xu, Thomas
Markovich, Florence Regol, and Yizhou Sun.

DSS-GNN models uncertainty-bearing node embeddings as random graph signals
expanded on two spectral axes: graph Fourier filters capture structural
variation, and a scalar orthogonal-polynomial (Wiener chaos) coordinate
captures latent stochastic variation. One representation supplies the
readouts for three tasks:

- **Calibrated prediction**: the quadrature-averaged predictive distribution
  over the chaos coordinate.
- **OOD detection**: the energy score of the mean (zeroth-order) logit.
- **Distribution-shift robustness**: the DSS branch used as a regularized
  spectral residual beside a deterministic encoder (**DSS-Hybrid**).

The model has two deployment modes, standalone DSS-GNN and DSS-Hybrid, and
both are evaluated on all three tasks.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). On a Linux
machine with a CUDA 12.4 driver:

```bash
uv sync
```

This installs PyTorch 2.5 (CUDA 12.4 wheels), PyTorch Geometric with the
`torch-scatter` and `torch-sparse` extensions, OGB, and the dependencies of
the vendored GOOD and G-DeltaUQ code. `requirements.txt` lists the same
packages for pip or conda users; the small graphs also run on CPU or Apple
MPS, slowly but workably.

Run the unit tests with

```bash
uv run pytest tests
```

## Data

Dataset directories are not tracked. The fixed evaluation splits are
included because they cannot be regenerated: `splits/` (geom-gcn style,
per-split `.npz`), `new_data2/splits/` (LINKX style `.npy`), and
`new_data/` (raw geom-gcn WebKB and Wikipedia graphs). To obtain the rest:

- `data/`, `data_pyg/`: created automatically. PyG (Planetoid, Amazon,
  Coauthor, Twitch, HeterophilousGraphDataset) and OGB (ogbn-arxiv) download
  on first use.
- `new_data2/*.npz` (roman_empire, amazon_ratings, minesweeper, questions,
  tolokers, filtered chameleon and squirrel, texas_4_classes): download from
  the `data/` directory of
  [yandex-research/heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs)
  into `new_data2/`.
- `new_data2/genius.mat`, `deezer-europe.mat`, `facebook100/`: from the
  `data/` directory of
  [CUAI/Non-Homophily-Large-Scale](https://github.com/CUAI/Non-Homophily-Large-Scale).
- `new_data2/twitch/`: optional local copy; the OOD loader falls back to the
  PyG Twitch download if it is absent.
- `GOOD_clean/storage/`: downloaded lazily by the GOOD package the first time
  `run_dssgnn_good.py` loads a dataset (about 23 GB for the full benchmark).

## Reproducing the paper

| Result | Entry point |
|---|---|
| Table 1, accuracy and Brier over 10 splits | `scripts/run_best_configs_acc_ece.py` (per-dataset configurations in `DSSGNN_BEST_CONFIG`) |
| Chaos-order, regularization, and quadrature ablations | `dssgnn_training.py` with `--P`, `--lambda_reg`, `--S` |
| Selected-configuration table with validation Brier | `scripts/reruns/` (job lists, launcher, parsers) |
| Tables 2 and 3, OOD detection (GNNSafe protocol) | `run_ood_gnnsafe.py` |
| Table 4, GOOD concept shift | `run_dssgnn_good.py` |
| DSS-Hybrid and its GCN base under the calibration protocol | `run_hybrid_calibration.py` |
| GCN, GAT, MC-dropout, and Deep Ensemble calibration baselines | `run_simple_baselines.py` |
| G-DeltaUQ baseline | `scripts/run_best_configs_acc_ece.py --model gduq`, `run_gduq_standalone.py` |

Examples:

```bash
# Table 1: DSS-GNN at the selected configuration of one dataset, 10 splits
python scripts/run_best_configs_acc_ece.py --dataset cora --model dssgnn --runs 10

# The same protocol with explicit hyperparameters
python dssgnn_training.py --dataset cora --runs 10 --P 1 --lambda_reg 0.01 --S 4

# DSS-Hybrid and its identically trained GCN base, calibration protocol
python run_hybrid_calibration.py --dataset roman-empire --runs 10
python run_hybrid_calibration.py --dataset roman-empire --runs 10 --model gcn

# MC-dropout (M=20) and Deep Ensemble (M=5) GCN baselines
python run_simple_baselines.py --model gcn --dataset cora --runs 10 --mc_samples 20
python run_simple_baselines.py --model gcn --dataset cora --runs 10 --ensemble_size 5

# OOD detection: DSS-Hybrid with the GNNSafe++ objective and score propagation
python run_ood_gnnsafe.py --dataset cora --ood_type structure --method gnnsafe \
    --backbone gcn_dssres --use_bn --use_reg --use_prop --runs 3
# Standalone DSS-GNN with per-chaos-channel BatchNorm
python run_ood_gnnsafe.py --dataset cora --ood_type structure --method gnnsafe \
    --backbone dssgnn --dss_bn --use_prop --runs 3
# Baselines on the same GCN backbone: graph_ebm, mc_dropout, ensemble, gduq
python run_ood_gnnsafe.py --dataset cora --ood_type structure --method mc_dropout \
    --backbone gcn --use_bn --use_prop --runs 3

# GOOD concept shift: DSS-Hybrid with standard ERM training
python run_dssgnn_good.py --config_path GOOD_clean/configs/GOOD_configs/GOODCora/word/concept/ERM.yaml \
    --model gcn_dssres --seed 42
# Standalone DSS-GNN on GOOD
python run_dssgnn_good.py --config_path GOOD_clean/configs/GOOD_configs/GOODCora/word/concept/ERM.yaml \
    --model dssgnn --layers 2 --normalize_features auto --dss_bn --seed 42
```

Every script prints `--help` with the full list of options. Hyperparameter
sweeps, log parsers, and figure scripts are in `scripts/`; the
`scripts/reruns/` job lists reproduce the per-dataset selection and
Graph-EBM re-runs reported in the appendix (`make_selection_jobs.py` writes
a TSV of commands, `launch_jobs.py` runs it on the given GPUs, and
`parse_selection_logs.py` collects the results).

## Repository layout

- `dssgnn/`: the model package. `model.py` holds `DSSGNN` and `DSSGNNTFE`,
  `dsconv_tfe_propfirst.py` the propagate-first variant, `chaos.py` the
  Hermite basis and Gauss-Hermite quadrature, `chebyshev.py` the polynomial
  graph filters, and `dssgnn_gduq.py` the G-DeltaUQ integration. Per-channel
  BatchNorm is opt-in (`use_bn`) and off by default.
- `dssgnn_training.py`, `tfe_utils.py`: the calibration protocol (10 fixed
  splits) and the shared data loading.
- `run_ood_gnnsafe.py`, `gnnsafe_ood/`: the OOD detection pipeline (GNNSafe
  benchmark: Cora, Amazon-Photo, and Coauthor-CS with structure, feature,
  and label shifts; cross-graph Twitch and Arxiv), with the Graph-EBM,
  MC-dropout, Deep Ensemble, and G-DeltaUQ baselines. `docs/scores.md`
  describes the available OOD scores.
- `run_dssgnn_good.py`, `GOOD_clean/`: the GOOD distribution-shift
  benchmark and its runner.
- `run_hybrid_calibration.py`, `run_simple_baselines.py`: DSS-Hybrid versus
  its base, and the standard GNN calibration baselines.
- `gduq/`, `external/graph-ebm/`, `tfe_*.py`: vendored baseline code
  (G-DeltaUQ, Graph-EBM, TFE-GNN); see `THIRD_PARTY.md`.
- `scripts/`: sweeps, parsers, plotting, and the re-run job lists.
- `tests/`: unit tests for the filters, chaos expansion, layers, model,
  OOD scores, and baselines.

## Citation

```bibtex
@inproceedings{xu2026dssgnn,
  title     = {A Unified Uncertainty Representation for Graph Neural Networks via Doubly-Spectral Stochastic Expansion},
  author    = {Xu, Fred and Markovich, Thomas and Regol, Florence and Sun, Yizhou},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
