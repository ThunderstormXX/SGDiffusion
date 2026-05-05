#!/bin/bash
# ==============================================================================
# MNIST Many-SGD Unified Runner
# ==============================================================================
# Usage:
#   ./run_mnist_manysgd.sh CONFIG_PATH [STAGE]
#
# STAGE:
#   0 = Run experiment (all tunnels)
#   1 = Visualization only
#   2 = Aggregation only
#   3 = Visualization + Aggregation
#   (omit) = Run ALL stages
#
# Examples:
#   ./run_mnist_manysgd.sh src/scripts/exp5/configs/mnist_manysgd/small_cartesian.json
#   ./run_mnist_manysgd.sh src/scripts/exp5/configs/mnist_manysgd/medium_cartesian.json 3
#
# Environment variables:
#   DEBUG=1  - Show tqdm for each training step
#   DEVICE=cuda - Use GPU
# ==============================================================================

set -e

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 CONFIG_PATH [STAGE]"
    echo ""
    echo "STAGE:  0=experiment, 1=visualize, 2=aggregate, 3=viz+agg"
    echo "        (if omitted: run ALL stages)"
    exit 1
fi

CONFIG_ARG=$1
STAGE="${2:-all}"
DEVICE="${DEVICE:-cpu}"
DEBUG="${DEBUG:-0}"

# Resolve config file path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${CONFIG_ARG}" = /* ]]; then
    CONFIG_FILE="${CONFIG_ARG}"
else
    CONFIG_FILE="${SCRIPT_DIR}/../../../../${CONFIG_ARG}"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

CONFIG_BASENAME="$(basename "${CONFIG_FILE}")"
CONFIG_NAME="${CONFIG_BASENAME%.json}"

# Parse JSON config with Python (portable)
read_json() {
    python3 -c "import json; d=json.load(open('$CONFIG_FILE')); print($1)"
}

# Read experiment info from JSON
EXPERIMENT_NAME=$(read_json "d['experiment_name']")
DESCRIPTION=$(read_json "d['description']")
NUM_TUNNELS=$(read_json "len(d['tunnels'])")

# Read data config
SAMPLE_SIZE=$(read_json "d['data'].get('sample_size', 6400)")
REPLACEMENT=$(read_json "str(d['data'].get('replacement', True)).lower()")

# Debug flag
if [ "$DEBUG" = "1" ]; then
    DEBUG_FLAG="--debug"
else
    DEBUG_FLAG=""
fi

# Output directory
OUTPUT_DIR="src/scripts/exp5/exp_results/mnist_manysgd/${CONFIG_NAME}"
AGGREGATED_DIR="${OUTPUT_DIR}/aggregated"

# Print header
REGIME_UPPER=$(echo "$CONFIG_NAME" | tr '[:lower:]' '[:upper:]')
echo "=============================================="
echo "MNIST MANY-SGD: ${REGIME_UPPER}"
echo "=============================================="
echo "  Config:  ${CONFIG_FILE}"
echo "  Stage:   ${STAGE} (0=exp, 1=viz, 2=agg, 3=viz+agg, all=full)"
echo "  Device:  ${DEVICE}"
echo "  Output:  ${OUTPUT_DIR}"
echo ""
echo "  ${DESCRIPTION}"
echo ""
echo "  Data: ${SAMPLE_SIZE} samples, replacement=${REPLACEMENT}"
echo ""

# Print tunnel info
for i in $(seq 0 $((NUM_TUNNELS - 1))); do
    TUNNEL_DESC=$(read_json "d['tunnels'][$i].get('description', 'Tunnel $i')")
    TUNNEL_NRUNS=$(read_json "d['tunnels'][$i].get('n_runs', 1)")
    TUNNEL_N_INIT=$(read_json "d['tunnels'][$i].get('n_initial_weights', 1)")
    if [ "$TUNNEL_N_INIT" -gt 1 ]; then
        echo "  Tunnel ${i}: ${TUNNEL_DESC} (${TUNNEL_N_INIT} init × ${TUNNEL_NRUNS} runs)"
    else
        echo "  Tunnel ${i}: ${TUNNEL_DESC} (${TUNNEL_NRUNS} runs)"
    fi
done

echo "=============================================="

# ==============================================================================
# STAGE 0: Run Experiment
# ==============================================================================

run_experiment() {
    echo ""
    echo "🚀 RUNNING EXPERIMENT"
    echo "=============================================="
    
    # Run each tunnel
    for i in $(seq 0 $((NUM_TUNNELS - 1))); do
        TUNNEL_DESC=$(read_json "d['tunnels'][$i].get('description', 'Tunnel $i')")
        TUNNEL_NRUNS=$(read_json "d['tunnels'][$i].get('n_runs', 1)")
        TUNNEL_N_INIT=$(read_json "d['tunnels'][$i].get('n_initial_weights', 1)")
        
        echo ""
        echo "▶ Tunnel ${i}: ${TUNNEL_DESC}"
        
        # Build command with optional n_initial_weights
        CMD="python -m src.scripts.exp5.run_tunnel \
            --config '${CONFIG_FILE}' \
            --tunnel ${i} \
            --n_runs ${TUNNEL_NRUNS} \
            --output_dir '${OUTPUT_DIR}' \
            --device '${DEVICE}'"
        
        # Add n_initial_weights only if > 1
        if [ "$TUNNEL_N_INIT" -gt 1 ]; then
            CMD="${CMD} --n_initial_weights ${TUNNEL_N_INIT}"
        fi
        
        # Add debug flag if set
        if [ -n "$DEBUG_FLAG" ]; then
            CMD="${CMD} ${DEBUG_FLAG}"
        fi
        
        # Execute
        eval $CMD
    done
    
    echo ""
    echo "✅ EXPERIMENT COMPLETE"
}

# ==============================================================================
# STAGE 1: Visualization
# ==============================================================================

run_visualization() {
    echo ""
    echo "📊 VISUALIZATION"
    echo "=============================================="
    python -m src.scripts.exp5.visualize_tunnel \
        --exp_dir "${OUTPUT_DIR}" \
        --output_dir "${AGGREGATED_DIR}"
}

# ==============================================================================
# STAGE 2: Aggregation
# ==============================================================================

run_aggregation() {
    echo ""
    echo "📦 AGGREGATION"
    echo "=============================================="
    python -m src.scripts.exp5.aggregate_tunnel \
        --exp_dir "${OUTPUT_DIR}" \
        --output_dir "${AGGREGATED_DIR}"
}

# ==============================================================================
# Execute based on STAGE
# ==============================================================================

case $STAGE in
    0)
        run_experiment
        ;;
    1)
        run_visualization
        ;;
    2)
        run_aggregation
        ;;
    3)
        run_visualization
        run_aggregation
        ;;
    all)
        run_experiment
        run_visualization
        run_aggregation
        ;;
    *)
        echo "ERROR: Invalid stage: $STAGE (must be 0, 1, 2, 3, or omit for all)"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "✅ DONE"
echo "=============================================="
