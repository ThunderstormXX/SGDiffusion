#!/bin/bash
set -euo pipefail

SETUP_NUM=${1:-1}

THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../../../.." && pwd)"

SETUP_DIR="$REPO_ROOT/src/scripts/exp2/bash"
PYTHON_DIR="$REPO_ROOT/src/scripts/exp2/python_scripts"
RESULTS_BASE="$REPO_ROOT/src/scripts/exp2/exp_results"
RESULTS_DIR="${RESULTS_BASE}/setup${SETUP_NUM}"

CONFIG="${SETUP_DIR}/setup${SETUP_NUM}/config.sh"
if [ -f "$CONFIG" ]; then
  source "$CONFIG"
else
  echo "Config not found for setup${SETUP_NUM}! ($CONFIG)"
  exit 1
fi

export PYTHONPATH="$REPO_ROOT"

echo "Running Validation Loss Distribution Analysis for setup ${SETUP_NUM}..."

# Find SGD weight files
WEIGHT_FILES=""
for lr in $(echo "${LRS_LIST}" | tr ',' ' '); do
  WEIGHT_FILE="${RESULTS_DIR}/sgd_weights_lr${lr}.pt"
  if [ -f "${WEIGHT_FILE}" ]; then
    if [ -z "${WEIGHT_FILES}" ]; then
      WEIGHT_FILES="${WEIGHT_FILE}"
    else
      WEIGHT_FILES="${WEIGHT_FILES},${WEIGHT_FILE}"
    fi
  else
    echo "Warning: Weight file not found: ${WEIGHT_FILE}"
  fi
done

if [ -z "${WEIGHT_FILES}" ]; then
  echo "No weight files found for analysis!"
  exit 1
fi

python "$PYTHON_DIR/val_loss_distribution.py" \
  --weights_files "${WEIGHT_FILES}" \
  --results_dir "${RESULTS_DIR}" \
  --device "${DEVICE}" \
  --n_samples 50

echo "Validation Loss Distribution Analysis completed for setup ${SETUP_NUM}!"

