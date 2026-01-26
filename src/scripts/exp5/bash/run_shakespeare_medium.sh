#!/bin/bash
# =============================================================================
# SHAKESPEARE MEDIUM: Standard experiment
# =============================================================================
# Dataset: Shakespeare (NanoGPT)
# N_RUNS: 20
# Stage 1 (SGD): 1000 steps, track WEIGHTS
# Stage 2 (GD):  100 epochs, track METRICS
# Stage 3 (SGD): 1000 steps, track WEIGHTS + HESSIANS
#
# Usage:
#   ./run_shakespeare_medium.sh       # Run all steps (0, 1, 2)
#   ./run_shakespeare_medium.sh 0     # Run only step 0 (experiment)
#   ./run_shakespeare_medium.sh 1     # Run only step 1 (visualize)
#   ./run_shakespeare_medium.sh 2     # Run only step 2 (aggregate)
#   ./run_shakespeare_medium.sh 0 1   # Run steps 0 and 1
#
# Device:
#   DEVICE=cuda ./run_shakespeare_medium.sh  # Run on CUDA
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Configuration
PRESET="shakespeare_medium"
N_RUNS=20
BASE_SEED=42
OUTPUT_DIR="src/scripts/exp5/exp_results/shakespeare_medium"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"

# Parse steps to run
if [ $# -eq 0 ]; then
    STEPS=(0 1 2)
else
    STEPS=("$@")
fi

echo "=============================================="
echo "SHAKESPEARE MEDIUM"
echo "=============================================="
echo "  Dataset: Shakespeare (NanoGPT)"
echo "  N_RUNS:  ${N_RUNS}"
echo "  Device:  ${DEVICE}"
echo "  Steps:   ${STEPS[*]}"
echo ""
echo "Stages:"
echo "  1. SGD: 1000 steps  (track: weights)"
echo "  2. GD:  100 epochs  (track: metrics)"
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
echo "SHAKESPEARE MEDIUM COMPLETE!"
echo "=============================================="
echo "Results: ${OUTPUT_DIR}/"
echo "=============================================="
