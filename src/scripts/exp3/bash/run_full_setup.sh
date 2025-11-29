#!/bin/bash
set -euo pipefail

# usage: ./run_full_setup.sh <SETUP_NUM>
SETUP_NUM=${1:-1}

# Get the path to this file and the partial setup script
THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARTIAL_SH="${THIS_DIR}/run_partial_setup.sh"

if [ ! -f "${PARTIAL_SH}" ]; then
  echo "run_partial_setup.sh not found at: ${PARTIAL_SH}"
  exit 1
fi

# Make the script executable just in case
chmod +x "${PARTIAL_SH}" 2>/dev/null || true

echo "=== Full setup ${SETUP_NUM}: start ==="

# Get the experiment directory from the current path
EXP_DIR="$(basename "$(dirname "$THIS_DIR")")"
echo "Running in experiment directory: $EXP_DIR"

for SCRIPT_NUM in 1 2 3 4 5; do
  echo "--- Running step ${SCRIPT_NUM} for setup ${SETUP_NUM} ---"
  "${PARTIAL_SH}" "${SETUP_NUM}" "${SCRIPT_NUM}"
done

echo "=== Full setup ${SETUP_NUM}: done ==="

