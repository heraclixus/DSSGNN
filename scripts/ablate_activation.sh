#!/usr/bin/env bash
# Ablation: activation function (relu vs gelu vs tanh) x chaos order P
# Tests whether smooth activations improve calibration or P-scaling behavior.
#
# Datasets: cora (homophilous), texas (small heterophilous), minesweeper (large heterophilous)
# P values: 0, 1, 2, 3, 4
# Activations: relu, gelu, tanh

set -euo pipefail

DATASETS="cora texas minesweeper"
P_VALUES="0 1 2 3 4"
ACT_FNS="relu gelu tanh"
RUNS=10
EPOCHS=1000
PATIENCE=200
LAMBDA_REG=0.01
S=4

OUTDIR="results_local/ablation_activation"
mkdir -p "$OUTDIR"

for dataset in $DATASETS; do
    for act_fn in $ACT_FNS; do
        for P in $P_VALUES; do
            tag="${dataset}_${act_fn}_P${P}"
            logfile="${OUTDIR}/${tag}.log"
            echo "=== $tag ==="
            python dssgnn_training.py \
                --dataset "$dataset" \
                --P "$P" \
                --S "$S" \
                --act_fn "$act_fn" \
                --lambda_reg "$LAMBDA_REG" \
                --runs "$RUNS" \
                --epochs "$EPOCHS" \
                --patience "$PATIENCE" \
                2>&1 | tee "$logfile"
        done
    done
done

echo "All done. Results in $OUTDIR/"
