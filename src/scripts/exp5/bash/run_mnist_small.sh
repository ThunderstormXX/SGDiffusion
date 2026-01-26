#!/bin/bash
# =============================================================================
# MNIST SMALL: Quick testing experiment
# =============================================================================
# Dataset: MNIST (FlexibleMLP)
# N_RUNS: 5
# Stage 1 (SGD): 100 steps,  track WEIGHTS
# Stage 2 (GD):  10 epochs,  track METRICS
# Stage 3 (SGD): 1000 steps, track WEIGHTS + HESSIANS
#
# Usage:
#   ./run_mnist_small.sh              # Run all steps (0, 1, 2)
#   ./run_mnist_small.sh 0            # Run only step 0 (experiment)
#   ./run_mnist_small.sh 1            # Run only step 1 (visualize)
#   ./run_mnist_small.sh 2            # Run only step 2 (aggregate)
#   ./run_mnist_small.sh 0 1          # Run steps 0 and 1
#   ./run_mnist_small.sh 1 2          # Run steps 1 and 2
#   ./run_mnist_small.sh 0 1 2        # Run all steps
#
# Device:
#   DEVICE=cuda ./run_mnist_small.sh  # Run on CUDA
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Configuration
PRESET="mnist_small"
N_RUNS=5
BASE_SEED=42
OUTPUT_DIR="src/scripts/exp5/exp_results/mnist_small"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"

# Parse steps to run
if [ $# -eq 0 ]; then
    STEPS=(0 1 2)  # Run all steps by default
else
    STEPS=("$@")
fi

echo "=============================================="
echo "MNIST SMALL"
echo "=============================================="
echo "  Dataset: MNIST (FlexibleMLP)"
echo "  N_RUNS:  ${N_RUNS}"
echo "  Device:  ${DEVICE}"
echo "  Steps:   ${STEPS[*]}"
echo ""
echo "Stages:"
echo "  1. SGD: 100 steps   (track: weights)"
echo "  2. GD:  10 epochs   (track: metrics)"
echo "  3. SGD: 1000 steps  (track: weights + hessians)"
echo "=============================================="

cd "${PROJECT_ROOT}"

# Source step functions
source "${SCRIPT_DIR}/steps/step0_experiment.sh"
source "${SCRIPT_DIR}/steps/step1_visualize.sh"
source "${SCRIPT_DIR}/steps/step2_aggregate.sh"

# Run requested steps
for step in "${STEPS[@]}"; do
    case $step in
        0)
            run_experiment "${PRESET}" "${N_RUNS}" "${BASE_SEED}" "${OUTPUT_DIR}" "${DEVICE}" "${DTYPE}"
            ;;
        1)
            run_visualize "${OUTPUT_DIR}" "${N_RUNS}" "${BASE_SEED}"
            ;;
        2)
            run_aggregate "${OUTPUT_DIR}" "${N_RUNS}" "${BASE_SEED}"
            ;;
        *)
            echo "ERROR: Unknown step: $step (valid: 0, 1, 2)"
            exit 1
            ;;
    esac
done

echo ""
echo "=============================================="
echo "MNIST SMALL COMPLETE!"
echo "=============================================="
echo "Results: ${OUTPUT_DIR}/"
echo "Plots:   ${OUTPUT_DIR}/aggregated/"
echo "=============================================="
