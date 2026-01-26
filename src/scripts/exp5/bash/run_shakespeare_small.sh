#!/bin/bash
# =============================================================================
# SHAKESPEARE SMALL: Quick testing experiment
# =============================================================================
# Dataset: Shakespeare (NanoGPT)
# N_RUNS: 5
# Stage 1 (SGD): 100 steps,  track WEIGHTS
# Stage 2 (GD):  10 epochs,  track METRICS
# Stage 3 (SGD): 1000 steps, track WEIGHTS + HESSIANS
#
# Usage:
#   ./run_shakespeare_small.sh        # Run all steps (0, 1, 2)
#   ./run_shakespeare_small.sh 0      # Run only step 0 (experiment)
#   ./run_shakespeare_small.sh 1      # Run only step 1 (visualize)
#   ./run_shakespeare_small.sh 2      # Run only step 2 (aggregate)
#   ./run_shakespeare_small.sh 0 1    # Run steps 0 and 1
#   ./run_shakespeare_small.sh 1 2    # Run steps 1 and 2 (reuse experiment)
#
# Device:
#   DEVICE=cuda ./run_shakespeare_small.sh  # Run on CUDA
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Configuration
PRESET="shakespeare_small"
N_RUNS=5
BASE_SEED=42
OUTPUT_DIR="src/scripts/exp5/exp_results/shakespeare_small"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"

# Parse steps to run
if [ $# -eq 0 ]; then
    STEPS=(0 1 2)
else
    STEPS=("$@")
fi

echo "=============================================="
echo "SHAKESPEARE SMALL"
echo "=============================================="
echo "  Dataset: Shakespeare (NanoGPT)"
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

source "${SCRIPT_DIR}/steps/step0_experiment.sh"
source "${SCRIPT_DIR}/steps/step1_visualize.sh"
source "${SCRIPT_DIR}/steps/step2_aggregate.sh"

for step in "${STEPS[@]}"; do
    case $step in
        0) run_experiment "${PRESET}" "${N_RUNS}" "${BASE_SEED}" "${OUTPUT_DIR}" "${DEVICE}" "${DTYPE}" ;;
        1) run_visualize "${OUTPUT_DIR}" "${N_RUNS}" "${BASE_SEED}" ;;
        2) run_aggregate "${OUTPUT_DIR}" "${N_RUNS}" "${BASE_SEED}" ;;
        *) echo "ERROR: Unknown step: $step"; exit 1 ;;
    esac
done

echo ""
echo "=============================================="
echo "SHAKESPEARE SMALL COMPLETE!"
echo "=============================================="
echo "Results: ${OUTPUT_DIR}/"
echo "Plots:   ${OUTPUT_DIR}/aggregated/"
echo "=============================================="
