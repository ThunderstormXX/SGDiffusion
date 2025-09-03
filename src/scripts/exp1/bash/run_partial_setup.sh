#!/bin/bash
set -euo pipefail

SETUP_NUM=${1:-1}
SCRIPT_NUM=${2:-1}

# --- вычисляем корень репозитория относительно этого файла ---
THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# THIS_DIR = .../src/scripts/exp1/bash
REPO_ROOT="$(cd "$THIS_DIR/../../../.." && pwd)"

SETUP_DIR="$REPO_ROOT/src/scripts/exp1/bash"
PYTHON_DIR="$REPO_ROOT/src/scripts/exp1/python_scripts"

# было:
# RESULTS_DIR="$REPO_ROOT/src/scripts/exp_results"
# стало (добавили постфикс по сетапу):
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

case $SCRIPT_NUM in
  1)
    echo "Running 1. SGD Training... -> ${RESULTS_DIR}"
    python "$PYTHON_DIR/train_sgd.py" \
      --dataset_train "${DATASET_TRAIN}" \
      --dataset_val "${DATASET_VAL}" \
      --model "${MODEL}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --seed "${SEED}" \
      --epochs "${EPOCHS_SGD}" \
      --results_dir "${RESULTS_DIR}"
    ;;
  2)
    echo "Running 2. GD Training... -> ${RESULTS_DIR}"
    python "$PYTHON_DIR/train_gd.py" \
      --dataset_train "${DATASET_TRAIN}" \
      --dataset_val "${DATASET_VAL}" \
      --model "${MODEL}" \
      --lr "${LR}" \
      --seed "${SEED}" \
      --epochs "${EPOCHS_GD}" \
      --checkpoint_in "initial_after_sgd.pt" \
      --results_dir "${RESULTS_DIR}"
    ;;
  3)
    echo "Running 3. Hessian trajectories... -> ${RESULTS_DIR}"
    python "$PYTHON_DIR/sgd_hessians.py" \
      --dataset_train "${DATASET_TRAIN}" \
      --dataset_val "${DATASET_VAL}" \
      --model "${MODEL}" \
      --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --lrs "${LRS_LIST}" \
      --steps "${HESSIAN_STEPS}" \
      --checkpoint_in "initial_after_sgd_and_gd.pt" \
      --results_dir "${RESULTS_DIR}"
    ;;
  4)
    echo "Running 4. Many runs (SGD)... -> ${RESULTS_DIR}"
    python "$PYTHON_DIR/sgd_many_runs.py" \
      --dataset_train "${DATASET_TRAIN}" \
      --model "${MODEL}" \
      --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --lrs "${LRS_LIST}" \
      --n_samples "${MANY_RUNS_SAMPLES}" \
      --steps "${MANY_RUNS_STEPS}" \
      --checkpoint_in "initial_after_sgd_and_gd.pt" \
      --results_dir "${RESULTS_DIR}"
    ;;
  5)
    echo "Running 5. Many runs (GD)... -> ${RESULTS_DIR}"
    python "$PYTHON_DIR/gd_many_runs.py" \
      --dataset_train "${DATASET_TRAIN}" \
      --model "${MODEL}" \
      --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --lrs "${LRS_LIST}" \
      --steps "${MANY_RUNS_STEPS}" \
      --checkpoint_in "initial_after_sgd_and_gd.pt" \
      --results_dir "${RESULTS_DIR}"
    ;;
  *)
    echo "Unknown script number: $SCRIPT_NUM"
    exit 1
    ;;
esac

echo "Completed script $SCRIPT_NUM for setup $SETUP_NUM!"
