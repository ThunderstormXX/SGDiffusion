#!/bin/bash
# =============================================================================
# STEP 1: Visualize results (weight trajectories)
# =============================================================================
# Creates 10 plots:
#   - 5 percentile plots (raw X-axis)
#   - 5 percentile plots (normalized X-axis: step × batch_size)
#
# Usage (called from parent script):
#   source step1_visualize.sh
#   run_visualize "$OUTPUT_DIR" "$N_RUNS" "$BASE_SEED"
# =============================================================================

run_visualize() {
    local OUTPUT_DIR="$1"
    local N_RUNS="$2"
    local BASE_SEED="$3"
    
    echo ""
    echo "=============================================="
    echo "STEP 1: VISUALIZING RESULTS"
    echo "=============================================="
    echo "  Output:    ${OUTPUT_DIR}"
    echo "  N_runs:    ${N_RUNS}"
    echo "  Base seed: ${BASE_SEED}"
    echo "=============================================="
    
    # Build list of run directories
    RUN_DIRS=""
    for i in $(seq 0 $((N_RUNS - 1))); do
        SEED=$((BASE_SEED + i))
        RUN_DIRS="${RUN_DIRS} ${OUTPUT_DIR}/run_seed${SEED}"
    done
    
    # Run visualization
    python -m src.scripts.exp5.visualize_many_runs \
        --run_dirs ${RUN_DIRS} \
        --output_dir "${OUTPUT_DIR}/aggregated" \
        --percentiles 0 20 40 60 80
    
    echo ""
    echo "=============================================="
    echo "STEP 1 COMPLETE: Visualizations saved"
    echo "  ${OUTPUT_DIR}/aggregated/"
    echo "=============================================="
}
