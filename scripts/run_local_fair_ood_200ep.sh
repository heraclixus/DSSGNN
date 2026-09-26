#!/usr/bin/env bash
set -euo pipefail

# Run a fully local, apples-to-apples OOD comparison at 200 epochs across the
# GNNSafe paper datasets.
#
# This script covers:
# - local GCN baselines: Energy, GNNSafe, Energy FT, GNNSafe++, Graph-EBM
# - local DSS hybrid baselines: Energy, GNNSafe, Energy FT, GNNSafe++
#
# Default coverage matches the GNNSafe paper:
# - Table 1 style: cora / amazon-photo / coauthor-cs x structure / feature / label
# - Table 2 style: twitch / arxiv
#
# Environment overrides:
#   PYTHON_BIN=/path/to/python
#   RUNS=3
#   EPOCHS=200
#   CPU=1
#   DEVICE=0
#   TAG_PREFIX=fair_local_200ep
#   SKIP_EXISTING=1
#   EXTRA_ARGS="--verbose"
#   TARGETS="cora:structure amazon-photo:feature twitch"
#
# Examples:
#   bash scripts/run_local_fair_ood_200ep.sh
#   RUNS=1 CPU=1 TARGETS="cora:structure arxiv" bash scripts/run_local_fair_ood_200ep.sh
#   CPU=0 DEVICE=0 bash scripts/run_local_fair_ood_200ep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
else
  PYTHON="python"
fi

RUNS="${RUNS:-3}"
EPOCHS="${EPOCHS:-200}"
CPU="${CPU:-1}"
DEVICE="${DEVICE:-0}"
TAG_PREFIX="${TAG_PREFIX:-fair_local_200ep}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TARGETS="${TARGETS:-}"

CONFIGS=(
  gcn_energy
  gcn_gnnsafe
  gcn_energy_ft
  gcn_gnnsafepp
  gcn_graph_ebm
  gcn_dssres_energy
  gcn_dssres_gnnsafe
  gcn_dssres_energy_ft
  gcn_dssres_gnnsafepp
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
  local out_dir="results_local/ood_density/${dataset}-${ood_type}-${tag}"

  if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/summary.csv" ]]; then
    echo
    echo "============================================================"
    echo "Skipping dataset=${dataset} ood_type=${ood_type} tag=${tag}"
    echo "Existing summary found at ${out_dir}/summary.csv"
    echo "============================================================"
    return
  fi

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

run_target_spec() {
  local spec="$1"
  local dataset
  local ood_type

  if [[ "$spec" == *:* ]]; then
    dataset="${spec%%:*}"
    ood_type="${spec##*:}"
  else
    dataset="$spec"
    ood_type="structure"
  fi

  run_case "$dataset" "$ood_type" "${TAG_PREFIX}_${dataset}_${ood_type}_r${RUNS}_e${EPOCHS}"
}

if [[ -n "$TARGETS" ]]; then
  # shellcheck disable=SC2206
  TARGET_SPECS=( $TARGETS )
  for spec in "${TARGET_SPECS[@]}"; do
    run_target_spec "$spec"
  done
else
  # GNNSafe Table 1 style
  for dataset in cora amazon-photo coauthor-cs; do
    for ood_type in structure feature label; do
      run_case "$dataset" "$ood_type" "${TAG_PREFIX}_${dataset}_${ood_type}_r${RUNS}_e${EPOCHS}"
    done
  done

  # GNNSafe Table 2 style
  for dataset in twitch arxiv; do
    run_case "$dataset" "structure" "${TAG_PREFIX}_${dataset}_r${RUNS}_e${EPOCHS}"
  done
fi

echo
echo "Finished fair local OOD matrix."
echo "Results are under results_local/ood_density/*${TAG_PREFIX}*"
