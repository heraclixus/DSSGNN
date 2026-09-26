#!/usr/bin/env bash
set -euo pipefail

# Run an apples-to-apples local OOD comparison between GCN baselines and
# the new post-hoc Graph-EBM baseline across the GNNSafe paper datasets.
#
# Default coverage matches the GNNSafe paper:
# - Table 1: cora / amazon-photo / coauthor-cs with structure, feature, label OOD
# - Table 2: arxiv / twitch
#
# Environment overrides:
#   PYTHON_BIN=/path/to/python
#   RUNS=3
#   EPOCHS=30
#   CPU=1
#   DEVICE=0
#   TAG_PREFIX=graph_ebm_matrix
#   EXTRA_ARGS="--verbose"
#
# Example:
#   bash scripts/run_graph_ebm_gnnsafe_local.sh
#   RUNS=1 EPOCHS=10 CPU=1 bash scripts/run_graph_ebm_gnnsafe_local.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
else
  PYTHON="python"
fi

RUNS="${RUNS:-3}"
EPOCHS="${EPOCHS:-30}"
CPU="${CPU:-1}"
DEVICE="${DEVICE:-0}"
TAG_PREFIX="${TAG_PREFIX:-graph_ebm_matrix}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CONFIGS=(
  gcn_energy
  gcn_gnnsafe
  gcn_energy_ft
  gcn_gnnsafepp
  gcn_graph_ebm
)

COMMON_ARGS=(
  --runs "$RUNS"
  --epochs "$EPOCHS"
  --configs "${CONFIGS[@]}"
)

if [[ "$CPU" == "1" ]]; then
  COMMON_ARGS+=(--cpu)
else
  COMMON_ARGS+=(--device "$DEVICE")
fi

run_case() {
  local dataset="$1"
  local ood_type="$2"
  local tag="$3"

  echo
  echo "============================================================"
  echo "Running dataset=${dataset} ood_type=${ood_type} tag=${tag}"
  echo "============================================================"

  local -a cmd
  cmd=(
    "$PYTHON"
    scripts/local_ood_density_compare.py
    --dataset "$dataset"
    --ood_type "$ood_type"
    --tag "$tag"
    "${COMMON_ARGS[@]}"
  )

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local -a extra=( $EXTRA_ARGS )
    cmd+=("${extra[@]}")
  fi

  "${cmd[@]}"
}

# GNNSafe Table 1
for dataset in cora amazon-photo coauthor-cs; do
  for ood_type in structure feature label; do
    run_case "$dataset" "$ood_type" "${TAG_PREFIX}_${dataset}_${ood_type}_r${RUNS}_e${EPOCHS}"
  done
done

# GNNSafe Table 2
for dataset in arxiv twitch; do
  run_case "$dataset" "structure" "${TAG_PREFIX}_${dataset}_r${RUNS}_e${EPOCHS}"
done

echo
echo "Finished Graph-EBM GNNSafe-style local matrix."
echo "Results are under results_local/ood_density/*${TAG_PREFIX}*"
