#!/bin/bash
# Run node classification calibration experiments (ECE) for Table 1 of G-ΔUQ paper
# https://arxiv.org/html/2401.03350v1
#
# Datasets: GOODCora (degree), GOODWebKB (university), GOODCBAS (color)
# Shifts: concept, covariate
# Metrics: Accuracy (↑), ECE (↓) - Expected Calibration Error (norm L1)
#
# Prerequisites:
#   1. Install GOOD: pip install -e . (from GOOD repo) or have GOOD in PYTHONPATH
#   2. Train baseline models first: nodeclassification_baseline.py
#   3. Train G-ΔUQ models: gduq_nodeclassification.py
#
# Config paths assume GOOD is installed with configs at GOOD/configs/GOOD_configs/

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GDUQ_SRC="$(cd "$SCRIPT_DIR/../src" && pwd)"
CKPT_DIR="${CKPT_DIR:-$GDUQ_SRC/../ckpts}"
CONFIG_ROOT="${CONFIG_ROOT:-$GDUQ_SRC/../GOOD/configs/GOOD_configs}"

# If GOOD configs not found, try common locations
if [ ! -d "$CONFIG_ROOT" ]; then
    for cand in "$(python -c 'import GOOD; import os; print(os.path.dirname(GOOD.__file__))' 2>/dev/null)/configs/GOOD_configs" \
                "$GDUQ_SRC/../GOOD/configs/GOOD_configs"; do
        if [ -d "$cand" ]; then
            CONFIG_ROOT="$cand"
            break
        fi
    done
fi

export PYTHONPATH="$GDUQ_SRC:$PYTHONPATH"
cd "$GDUQ_SRC"

# Post-hoc calibration methods (Table 1: ✕, CAGCN, Dirichlet, ETS, GATS, IRM, Orderinvariant, Spline, VS)
UQ_NAMES="vanilla ets ts vs irm dirichlet spline orderinvariant cagcn gats"

# Datasets and domains from Table 1 (node classification)
# GOODCora: degree | GOODWebKB: university | GOODCBAS: color
DATASETS=("GOODCora:degree" "GOODWebKB:university" "GOODCBAS:color")
SHIFTS=("concept" "covariate")

echo "=== Node Classification Calibration Experiments (ECE) ==="
echo "Config root: $CONFIG_ROOT"
echo "Checkpoint dir: $CKPT_DIR"
echo ""

# --- 1. Baseline (No G-ΔUQ) with post-hoc calibration ---
echo ">>> Running BASELINE (No G-ΔUQ) evaluation..."
for ds_domain in "${DATASETS[@]}"; do
    IFS=':' read -r dataset domain <<< "$ds_domain"
    for shift in "${SHIFTS[@]}"; do
        config_path="$CONFIG_ROOT/$dataset/$domain/$shift/ERM.yaml"
        if [ ! -f "$config_path" ]; then
            echo "  [SKIP] Config not found: $config_path"
            continue
        fi
        ckpt="$CKPT_DIR/$dataset/baseline_${dataset}_${domain}_${shift}_GCN_0.ckpt"
        if [ ! -f "$ckpt" ]; then
            echo "  [SKIP] Checkpoint not found: $ckpt (train baseline first)"
            continue
        fi
        for uq in $UQ_NAMES; do
            echo "  Baseline | $dataset | $domain | $shift | $uq"
            python eval_posthoc/eval_nodeclassification_baseline.py \
                --config_path "$config_path" \
                --uq_name "$uq" \
                --ckpt_path "$ckpt" 2>/dev/null || true
        done
    done
done

# --- 2. G-ΔUQ with post-hoc calibration ---
echo ""
echo ">>> Running G-ΔUQ evaluation..."
for ds_domain in "${DATASETS[@]}"; do
    IFS=':' read -r dataset domain <<< "$ds_domain"
    for shift in "${SHIFTS[@]}"; do
        config_path="$CONFIG_ROOT/$dataset/$domain/$shift/ERM.yaml"
        if [ ! -f "$config_path" ]; then
            echo "  [SKIP] Config not found: $config_path"
            continue
        fi
        ckpt="$CKPT_DIR/$dataset/baseline_${dataset}_${domain}_${shift}_GCN_0.ckpt"
        if [ ! -f "$ckpt" ]; then
            echo "  [SKIP] Checkpoint not found: $ckpt"
            continue
        fi
        for uq in $UQ_NAMES; do
            echo "  G-ΔUQ | $dataset | $domain | $shift | $uq"
            python eval_posthoc/eval_nodeclassification_gduq.py \
                --config_path "$config_path" \
                --uq_name "$uq" \
                --ckpt_path "$ckpt" \
                --anchor_type "graph" \
                --num_anchors 10 2>/dev/null || true
        done
    done
done

echo ""
echo "=== Done. Results appended to logs.csv ==="
echo "For Table 1: use OOD 'test' split Accuracy and ECE (cal_err_l1)."
