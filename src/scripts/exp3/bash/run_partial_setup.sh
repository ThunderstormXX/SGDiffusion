#!/bin/bash
set -euo pipefail

SETUP_NUM=${1:-1}
SCRIPT_NUM=${2:-1}

# Calculate repository root relative to this file
THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../../../.." && pwd)"

# Get the experiment directory from the current path
EXP_DIR="$(basename "$(dirname "$THIS_DIR")")"

SETUP_DIR="$THIS_DIR"
PYTHON_DIR="$REPO_ROOT/src/scripts/$EXP_DIR/python_scripts"

# Results directory with experiment-specific path
RESULTS_BASE="$REPO_ROOT/src/scripts/$EXP_DIR/exp_results"
RESULTS_DIR="${RESULTS_BASE}/setup${SETUP_NUM}"

CONFIG="${SETUP_DIR}/setup${SETUP_NUM}/config.sh"
if [ -f "$CONFIG" ]; then
  # shellcheck source=/dev/null
  source "$CONFIG"
else
  echo "Config not found for setup${SETUP_NUM}! ($CONFIG)"
  exit 1
fi

# Important: Set PYTHONPATH to root for imports to work correctly
export PYTHONPATH="$REPO_ROOT"
mkdir -p "${RESULTS_DIR}"

case $SCRIPT_NUM in
  1)
    echo "Running 1. Training with multiple optimizers... -> ${RESULTS_DIR}"
    # Parse optimizers from config
    IFS=',' read -ra OPTIMIZER_LIST <<< "${OPTIMIZERS}"
    
    for opt in "${OPTIMIZER_LIST[@]}"; do
      echo "Training with optimizer: ${opt}"
      python "$PYTHON_DIR/train_sgd.py" \
        --dataset_train "${DATASET_TRAIN}" \
        --dataset_val "${DATASET_VAL}" \
        --model "${MODEL}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --seed "${SEED}" \
        --epochs "${EPOCHS_SGD}" \
        --optimizer "${opt}" \
        --optimizer_params "${OPTIMIZER_PARAMS}" \
        --results_dir "${RESULTS_DIR}/${opt}" \
        --data_loader "${DATALOADER}" \
        --device "${DEVICE}" \
        --sample_size "${SAMPLE_SIZE}" \
        --model_params "${MODEL_PARAMS}" \
        --model_params "${MODEL_PARAMS}" \
        --save_trajectory
    done
    ;;
    
  2)
    echo "Running 2. GD Training from optimizer checkpoints... -> ${RESULTS_DIR}"
    # Parse optimizers from config
    IFS=',' read -ra OPTIMIZER_LIST <<< "${OPTIMIZERS}"
    
    for opt in "${OPTIMIZER_LIST[@]}"; do
      echo "Running GD from ${opt} checkpoint"
      python "$PYTHON_DIR/train_gd.py" \
        --dataset_train "${DATASET_TRAIN}" \
        --dataset_val "${DATASET_VAL}" \
        --model "${MODEL}" \
        --lr "${LR}" \
        --seed "${SEED}" \
        --epochs "${EPOCHS_GD}" \
        --checkpoint_in "${RESULTS_DIR}/${opt}/initial_after_sgd.pt" \
        --results_dir "${RESULTS_DIR}/${opt}/gd" \
        --device "${DEVICE}" \
        --sample_size "${SAMPLE_SIZE}" \
        --model_params "${MODEL_PARAMS}"
    done
    ;;
    
  3)
    echo "Running 3. Valley exploration with SGD from GD checkpoints... -> ${RESULTS_DIR}"
    # Parse optimizers from config
    IFS=',' read -ra OPTIMIZER_LIST <<< "${OPTIMIZERS}"
    
    for opt in "${OPTIMIZER_LIST[@]}"; do
      echo "Exploring valley from ${opt} GD checkpoint"
      python "$PYTHON_DIR/valley_exploration.py" \
        --dataset_train "${DATASET_TRAIN}" \
        --dataset_val "${DATASET_VAL}" \
        --model "${MODEL}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --seed "${SEED}" \
        --steps "${VALLEY_STEPS}" \
        --checkpoint_in "${RESULTS_DIR}/${opt}/gd/initial_after_sgd_and_gd.pt" \
        --results_dir "${RESULTS_DIR}/${opt}/valley" \
        --data_loader "${DATALOADER}" \
        --device "${DEVICE}" \
        --sample_size "${SAMPLE_SIZE}" \
        --model_params "${MODEL_PARAMS}"
    done
    ;;
    
  4)
    echo "Running 4. Hessian analysis... -> ${RESULTS_DIR}"
    # Parse optimizers from config
    IFS=',' read -ra OPTIMIZER_LIST <<< "${OPTIMIZERS}"
    
    for opt in "${OPTIMIZER_LIST[@]}"; do
      echo "Analyzing Hessian for ${opt}"
      python "$PYTHON_DIR/sgd_hessians.py" \
        --dataset_train "${DATASET_TRAIN}" \
        --dataset_val "${DATASET_VAL}" \
        --model "${MODEL}" \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}" \
        --lrs "${LRS_LIST}" \
        --steps "${HESSIAN_STEPS}" \
        --checkpoint_in "${RESULTS_DIR}/${opt}/gd/initial_after_sgd_and_gd.pt" \
        --results_dir "${RESULTS_DIR}/${opt}/hessian" \
        --data_loader "${DATALOADER}" \
        --device "${DEVICE}" \
        --sample_size "${SAMPLE_SIZE}" \
        --model_params "${MODEL_PARAMS}"
    done
    ;;
    
  5)
    echo "Running 5. Many runs analysis... -> ${RESULTS_DIR}"
    # Parse optimizers from config
    IFS=',' read -ra OPTIMIZER_LIST <<< "${OPTIMIZERS}"
    
    for opt in "${OPTIMIZER_LIST[@]}"; do
      echo "Running SGD many runs analysis for ${opt}"
      python "$PYTHON_DIR/sgd_many_runs.py" \
        --dataset_train "${DATASET_TRAIN}" \
        --model "${MODEL}" \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}" \
        --lrs "${LRS_LIST}" \
        --n_samples "${MANY_RUNS_SAMPLES}" \
        --steps "${MANY_RUNS_STEPS}" \
        --checkpoint_in "${RESULTS_DIR}/${opt}/gd/initial_after_sgd_and_gd.pt" \
        --results_dir "${RESULTS_DIR}/${opt}/many_runs" \
        --data_loader "${DATALOADER}" \
        --device "${DEVICE}" \
        --sample_size "${SAMPLE_SIZE}" \
        --model_params "${MODEL_PARAMS}"
    done
    ;;
    
  6)
    echo "Running 6. Combining and visualizing losses... -> ${RESULTS_DIR}"
    # Parse optimizers from config
    IFS=',' read -ra OPTIMIZER_LIST <<< "${OPTIMIZERS}"
    
    echo "Combining losses for all optimizers"
    python "$PYTHON_DIR/combine_losses.py" \
      --results_dir "${RESULTS_DIR}" \
      --optimizer_list "${OPTIMIZERS}" \
      --output_dir "${RESULTS_DIR}/combined_plots"
    ;;
    
  *)
    echo "Unknown script number: $SCRIPT_NUM"
    exit 1
    ;;
esac

echo "Completed script $SCRIPT_NUM for setup $SETUP_NUM!"
