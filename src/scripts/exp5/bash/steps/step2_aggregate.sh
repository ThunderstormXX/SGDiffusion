#!/bin/bash
# =============================================================================
# STEP 2: Aggregate tensors from all runs
# =============================================================================
# Creates aggregated tensors:
#   - weights_{stage}.pt: (n_runs, n_steps, n_params)
#   - hessians_{stage}.pt: (n_runs, n_steps, n_params, n_params)
#   - grads_{stage}.pt: (n_runs, n_steps, n_params)
#   - {name}_tensor_info.txt: Shape and metadata
#
# Usage (called from parent script):
#   source step2_aggregate.sh
#   run_aggregate "$OUTPUT_DIR" "$N_RUNS" "$BASE_SEED"
# =============================================================================

run_aggregate() {
    local OUTPUT_DIR="$1"
    local N_RUNS="$2"
    local BASE_SEED="$3"
    
    echo ""
    echo "=============================================="
    echo "STEP 2: AGGREGATING TENSORS"
    echo "=============================================="
    echo "  Output:    ${OUTPUT_DIR}"
    echo "  N_runs:    ${N_RUNS}"
    echo "  Base seed: ${BASE_SEED}"
    echo "=============================================="
    
    python -m src.scripts.exp5.aggregate_tensors \
        --input_dir "${OUTPUT_DIR}" \
        --n_runs "${N_RUNS}" \
        --base_seed "${BASE_SEED}" \
        --output_dir "${OUTPUT_DIR}/aggregated"
    
    echo ""
    echo "=============================================="
    echo "STEP 2 COMPLETE: Tensors aggregated"
    echo "  ${OUTPUT_DIR}/aggregated/"
    echo "=============================================="
}
