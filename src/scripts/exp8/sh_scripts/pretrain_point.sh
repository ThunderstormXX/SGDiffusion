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
BATCH_SIZE="${BATCH_SIZE:-64}"
PRETRAIN_SGD_STEPS="${PRETRAIN_SGD_STEPS:-${PRETRAIN_STEPS:-10000}}"
PRETRAIN_GD_EPOCHS="${PRETRAIN_GD_EPOCHS:-500}"
LR="${LR:-0.1}"
HIDDEN_DIM="${HIDDEN_DIM:-48}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-1}"
INPUT_DOWNSAMPLE="${INPUT_DOWNSAMPLE:-14}"
DROPOUT="${DROPOUT:-0.0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-mps}"
CHECKPOINT_OUT="${CHECKPOINT_OUT:-$EXP_DIR/results/pretrained_points/pretrained_point.pt}"
PRETRAIN_FIGURES_DIR="${PRETRAIN_FIGURES_DIR:-$EXP_DIR/figures/pretrain}"
PRETRAIN_LOGS_DIR="${PRETRAIN_LOGS_DIR:-$EXP_DIR/results/pretrained_points}"

python "$EXP_DIR/src/pretrain.py" \
  --dataset "$DATASET" \
  --architecture "$ARCHITECTURE" \
  --train_size "$TRAIN_SIZE" \
  --val_size "$VAL_SIZE" \
  --test_size "$TEST_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --pretrain_sgd_steps "$PRETRAIN_SGD_STEPS" \
  --pretrain_gd_epochs "$PRETRAIN_GD_EPOCHS" \
  --lr "$LR" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
  --input_downsample "$INPUT_DOWNSAMPLE" \
  --dropout "$DROPOUT" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --checkpoint_out "$CHECKPOINT_OUT" \
  --figures_dir "$PRETRAIN_FIGURES_DIR" \
  --logs_dir "$PRETRAIN_LOGS_DIR" \
  "$@"
