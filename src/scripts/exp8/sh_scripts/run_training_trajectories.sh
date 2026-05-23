#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$THIS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATASET="${DATASET:-mnist}"
ARCHITECTURE="${ARCHITECTURE:-flexible_mlp}"
TRAIN_SIZE="${TRAIN_SIZE:-6400}"
VAL_SIZE="${VAL_SIZE:-100}"
TEST_SIZE="${TEST_SIZE:-100}"
RUNS="${RUNS:-10}"
SGD_ITERATIONS="${SGD_ITERATIONS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.1}"
HIDDEN_DIM="${HIDDEN_DIM:-48}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-1}"
INPUT_DOWNSAMPLE="${INPUT_DOWNSAMPLE:-14}"
DROPOUT="${DROPOUT:-0.0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-mps}"
CHECKPOINT_IN="${CHECKPOINT_IN:-}"

python "$EXP_DIR/src/run.py" \
  --dataset "$DATASET" \
  --architecture "$ARCHITECTURE" \
  --train_size "$TRAIN_SIZE" \
  --val_size "$VAL_SIZE" \
  --test_size "$TEST_SIZE" \
  --runs "$RUNS" \
  --sgd_iterations "$SGD_ITERATIONS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
  --input_downsample "$INPUT_DOWNSAMPLE" \
  --dropout "$DROPOUT" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --checkpoint_in "$CHECKPOINT_IN" \
  "$@"
