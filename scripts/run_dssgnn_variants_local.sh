#!/bin/bash
# Local test for DSS-GNN variants sweep (single dataset, single variant).
# Usage: ./scripts/run_dssgnn_variants_local.sh [dataset] [variant]
# Example: ./scripts/run_dssgnn_variants_local.sh cora dssgnn_sg

set -e
cd "$(dirname "$0")/.."
DATASET="${1:-cora}"
VARIANT="${2:-dssgnn}"

echo "Running dataset=${DATASET} variant=${VARIANT}"
python scripts/run_dssgnn_variants_sweep.py --dataset "$DATASET" --variant "$VARIANT" --runs 2
