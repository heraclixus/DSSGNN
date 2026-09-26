#!/bin/bash
# Run a single OOD experiment (for quick local testing).
# Examples:
#   ./scripts/run_ood_single.sh msp gcn cora structure
#   ./scripts/run_ood_single.sh msp dssgnn cora structure    # DSS-GNN
#   ./scripts/run_ood_single.sh msp tfe cora structure       # TFEGNN
#   ./scripts/run_ood_single.sh gduq gduq cora structure      # GDUQ
#   USE_PROP=1 USE_REG=1 ./scripts/run_ood_single.sh gnnsafe gcn cora structure  # GNNSafe++

set -euo pipefail

METHOD="${1:-msp}"
BACKBONE="${2:-gcn}"
DATASET="${3:-cora}"
OOD_TYPE="${4:-structure}"
RUNS="${5:-1}"
EPOCHS="${6:-50}"

cd "$(dirname "$0")/.."

EXTRA=()
if [[ "$METHOD" == "gnnsafe" ]]; then
  # Default: Energy (no prop, no reg)
  USE_PROP="${USE_PROP:-0}"
  USE_REG="${USE_REG:-0}"
  [[ "$USE_PROP" == "1" ]] && EXTRA+=(--use_prop)
  if [[ "$USE_REG" == "1" ]]; then
    EXTRA+=(--use_reg)
    case "$DATASET" in
      cora) EXTRA+=(--m_in -5 --m_out -1 --lamda 0.01) ;;
      amazon-photo) EXTRA+=(--m_in -9 --m_out -1 --lamda 0.1) ;;
      coauthor-cs) EXTRA+=(--m_in -5 --m_out -1 --lamda 0.1) ;;
      twitch) EXTRA+=(--m_in -5 --m_out -1 --lamda 0.1) ;;
      arxiv) EXTRA+=(--m_in -9 --m_out -2 --lamda 0.01) ;;
      *) EXTRA+=(--m_in -5 --m_out -1 --lamda 0.01) ;;
    esac
  fi
fi

OOD_ARG=()
[[ "$DATASET" =~ ^(cora|amazon-photo|coauthor-cs)$ ]] && OOD_ARG=(--ood_type "$OOD_TYPE")

BN=()
[[ "$DATASET" =~ ^(amazon-photo|amazon-computer|coauthor-cs|coauthor-physics)$ ]] && BN=(--use_bn)

python run_ood_gnnsafe.py \
  --method "$METHOD" \
  --backbone "$BACKBONE" \
  --dataset "$DATASET" \
  --mode detect \
  --runs "$RUNS" \
  --epochs "$EPOCHS" \
  "${OOD_ARG[@]}" \
  "${BN[@]}" \
  "${EXTRA[@]}"
