#!/bin/bash
# =============================================================================
# STEP 0: Run experiment (all 3 stages sequentially for all runs)
# =============================================================================
# This step runs:
#   Stage 1 (SGD) for all seeds
#   Stage 2 (GD) for all seeds (using stage 1 checkpoints)
#   Stage 3 (SGD) for all seeds (using stage 2 checkpoints)
#
# Usage (called from parent script):
#   source step0_experiment.sh
#   run_experiment "$PRESET" "$N_RUNS" "$BASE_SEED" "$OUTPUT_DIR" "$DEVICE" "$DTYPE"
# =============================================================================

run_experiment() {
    local PRESET="$1"
    local N_RUNS="$2"
    local BASE_SEED="$3"
    local OUTPUT_DIR="$4"
    local DEVICE="$5"
    local DTYPE="$6"
    
    echo ""
    echo "=============================================="
    echo "STEP 0: RUNNING EXPERIMENT"
    echo "=============================================="
    echo "  Preset:    ${PRESET}"
    echo "  N_runs:    ${N_RUNS}"
    echo "  Base seed: ${BASE_SEED}"
    echo "  Output:    ${OUTPUT_DIR}"
    echo "  Device:    ${DEVICE}"
    echo "=============================================="
    
    # Stage 1: SGD for all seeds
    echo ""
    echo ">>> Stage 1: SGD (${N_RUNS} runs)"
    python -m src.scripts.exp5.run_stage \
        --preset "${PRESET}" \
        --stage 1 \
        --n_runs "${N_RUNS}" \
        --base_seed "${BASE_SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        --device "${DEVICE}" \
        --dtype "${DTYPE}"
    
    # Stage 2: GD for all seeds (uses stage 1 checkpoints)
    echo ""
    echo ">>> Stage 2: GD (${N_RUNS} runs)"
    python -m src.scripts.exp5.run_stage \
        --preset "${PRESET}" \
        --stage 2 \
        --n_runs "${N_RUNS}" \
        --base_seed "${BASE_SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        --device "${DEVICE}" \
        --dtype "${DTYPE}"
    
    # Stage 3: SGD for all seeds (uses stage 2 checkpoints)
    echo ""
    echo ">>> Stage 3: SGD (${N_RUNS} runs)"
    python -m src.scripts.exp5.run_stage \
        --preset "${PRESET}" \
        --stage 3 \
        --n_runs "${N_RUNS}" \
        --base_seed "${BASE_SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        --device "${DEVICE}" \
        --dtype "${DTYPE}"
    
    echo ""
    echo "=============================================="
    echo "STEP 0 COMPLETE: Experiment finished"
    echo "=============================================="
}
