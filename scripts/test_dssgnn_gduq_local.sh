#!/bin/bash
# Quick local test of DS-GNN + GDUQ on small graph.
# Run from project root. Use signature env: conda activate signature
#
# Usage:
#   ./scripts/test_dssgnn_gduq_local.sh          # full comparison (Cora)
#   ./scripts/test_dssgnn_gduq_local.sh quick    # synthetic only (~5s)

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "quick" ]; then
  echo "=== Quick: synthetic graph only ==="
  python scripts/test_dssgnn_gduq_synthetic.py
  exit 0
fi

echo "=== 1. Synthetic graph (fast sanity check) ==="
python scripts/test_dssgnn_gduq_synthetic.py

echo ""
echo "=== 2. DS-GNN + GDUQ (Cora, 2 runs, 50 epochs) ==="
python run_dssgnn_gduq.py \
  --dataset cora \
  --runs 2 \
  --epochs 50 \
  --patience 20 \
  --num_anchors 3 \
  --hidden 64 \
  2>&1

echo ""
echo "=== 3. Vanilla DS-GNN (no anchor) ==="
python dssgnn_training.py \
  --dataset cora \
  --runs 2 \
  --epochs 50 \
  --patience 20 \
  2>&1 | tail -15

echo ""
echo "=== 4. GDUQ (GCN base) ==="
python run_gduq_standalone.py \
  --dataset cora \
  --runs 2 \
  --epochs 50 \
  --patience 20 \
  --num_anchors 3 \
  2>&1 | tail -10
