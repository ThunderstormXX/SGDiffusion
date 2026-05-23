#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$THIS_DIR/.." && pwd)"

CHECKPOINT_IN="${CHECKPOINT_IN:-$EXP_DIR/results/pretrained_points/pretrained_point.pt}"
FIGURES_DIR="${FIGURES_DIR:-$EXP_DIR/figures/from_pretrained_point}"
RESULTS_DIR="${RESULTS_DIR:-$EXP_DIR/results/from_pretrained_point}"

if [ ! -f "$CHECKPOINT_IN" ]; then
  echo "Checkpoint not found: $CHECKPOINT_IN"
  echo "Run: $EXP_DIR/sh_scripts/pretrain_point.sh"
  exit 1
fi

CHECKPOINT_IN="$CHECKPOINT_IN" \
  "$EXP_DIR/sh_scripts/run_training_trajectories.sh" \
  --figures_dir "$FIGURES_DIR" \
  --results_dir "$RESULTS_DIR" \
  "$@"
