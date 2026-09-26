# Node Classification Calibration Experiments (ECE)

This document describes how to run calibration experiments for node classification as in **Table 1** of the G-ΔUQ paper ([arxiv.org/html/2401.03350v1](https://arxiv.org/html/2401.03350v1)).

## Metrics

- **ECE (Expected Calibration Error)**: L1 norm, 100 bins (`MulticlassCalibrationError(norm='l1')`)
- **Accuracy**: Micro-averaged multiclass accuracy
- Reported on **OOD test** split (`test` in GOOD loaders) for concept and covariate shift

## Datasets (GOOD Benchmark)

| Dataset    | Domain     | Concept | Covariate |
|-----------|------------|--------|-----------|
| GOODCora  | degree     | ✓      | ✓         |
| GOODWebKB | university | ✓      | ✓         |
| GOODCBAS  | color      | ✓      | ✓         |

## Pipeline

### 1. Install GOOD

```bash
git clone https://github.com/divelab/GOOD.git && cd GOOD
pip install -e .
```

### 2. Train Baseline Models

```bash
cd gduq/src
# For each (dataset, domain, shift), e.g.:
python nodeclassification_baseline.py --config_path <path_to_GOOD>/configs/GOOD_configs/GOODCora/degree/concept/ERM.yaml
```

Checkpoints are saved to `ckpts/<dataset>/baseline_<dataset>_<domain>_<shift>_GCN_<seed>.ckpt`.

### 3. Train G-ΔUQ Models

```bash
python gduq_nodeclassification.py --config_path <config> --anchor_type graph --num_anchors 10
```

### 4. Run Calibration Evaluation

**Baseline (No G-ΔUQ)** with post-hoc calibration:

```bash
python eval_posthoc/eval_nodeclassification_baseline.py \
  --config_path <GOOD_config_path> \
  --uq_name vanilla \
  --ckpt_path <baseline_ckpt>
```

**G-ΔUQ** with post-hoc calibration:

```bash
python eval_posthoc/eval_nodeclassification_gduq.py \
  --config_path <GOOD_config_path> \
  --uq_name vanilla \
  --ckpt_path <baseline_ckpt> \
  --anchor_type graph \
  --num_anchors 10
```

### Post-hoc Calibration Methods (`--uq_name`)

| Name          | Paper Table 1 |
|---------------|----------------|
| vanilla       | ✕ (no calibration) |
| ets           | ETS            |
| ts            | TS             |
| vs            | VS             |
| irm           | IRM            |
| dirichlet     | Dirichlet      |
| spline        | Spline         |
| orderinvariant| Orderinvariant |
| cagcn         | CAGCN          |
| gats          | GATS           |

### 5. Output

Results are appended to `logs.csv` with columns:
`save_name, loader_key, acc, cal_err_l1, cal_err_l2, cal_err_max`

For Table 1, use **loader_key=test** (OOD test) and **cal_err_l1** as ECE.

## Batch Script

```bash
cd gduq/scripts
export CONFIG_ROOT=/path/to/GOOD/configs/GOOD_configs
export CKPT_DIR=/path/to/ckpts
bash run_node_calibration_experiments.sh
```

## Environment

- `GDUQ_DIST_CACHE`: Optional directory for GATS `dist_to_train` cache (default: `gduq/src/dist_to_train`)
- `CKPT_DIR`: Checkpoint root
- `CONFIG_ROOT`: GOOD config root (e.g. `GOOD/configs/GOOD_configs`)
