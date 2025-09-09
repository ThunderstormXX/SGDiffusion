#!/bin/bash

SETUP_NUM=${1:-1}

THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../../../.." && pwd)"

SETUP_DIR="$REPO_ROOT/src/scripts/exp1/bash"
PYTHON_DIR="$REPO_ROOT/src/scripts/exp1/python_scripts"

RESULTS_BASE="$REPO_ROOT/src/scripts/exp1/exp_results"
RESULTS_DIR="${RESULTS_BASE}/setup${SETUP_NUM}"

CONFIG="${SETUP_DIR}/setup${SETUP_NUM}/config.sh"
if [ -f "$CONFIG" ]; then
  # shellcheck source=/dev/null
  source "$CONFIG"
else
  echo "Config not found for setup${SETUP_NUM}! ($CONFIG)"
  exit 1
fi

# ВАЖНО: PYTHONPATH = корень, чтобы работало `from src.model import ...`
export PYTHONPATH="$REPO_ROOT"
mkdir -p "${RESULTS_DIR}"

echo "Running 4. Many runs... -> ${RESULTS_DIR}"
    python "$PYTHON_DIR/gd_many_runs.py" \
      --dataset_train "${DATASET_TRAIN}" \
      --model "${MODEL}" \
      --batch_size "${BATCH_SIZE}" \
      --sample_size "${SAMPLE_SIZE}" \
      --seed "${SEED}" \
      --lrs "${LRS_LIST}" \
      --steps "${MANY_RUNS_STEPS}" \
      --checkpoint_in "initial_after_sgd_and_gd.pt" \
      --results_dir "${RESULTS_DIR}" \
      --data_loader "${DATALOADER}" \
      --lr_scaling "${GD_SCALING}"