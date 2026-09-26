#!/bin/bash
# Run best configs locally (no SLURM). For quick tests or single-machine runs.
# Usage: ./scripts/run_best_configs_acc_ece_local.sh [dataset] [runs]
# Example: ./scripts/run_best_configs_acc_ece_local.sh cora 2

set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:-cora}"
RUNS="${2:-2}"

echo "Running best configs Acc+ECE: dataset=${DATASET} runs=${RUNS}"
for model in dssgnn tfe gduq; do
  echo "--- ${model} ---"
  python scripts/run_best_configs_acc_ece.py --dataset "${DATASET}" --model "${model}" --runs "${RUNS}" 2>&1 | tee -a results_local/logs/best_acc_ece_local_${DATASET}.out
done
echo "Done. Parse with: grep RESULT results_local/logs/best_acc_ece_local_${DATASET}.out"
