#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <setup_id>"
  exit 1
fi

SETUP_ID=$1
SETUP_DIR="src/scripts/exp1/bash/setup${SETUP_ID}"
CONFIG_FILE="${SETUP_DIR}/config.sh"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "[error] Config file not found: $CONFIG_FILE"
  exit 2
fi

# Подгружаем переменные
source "$CONFIG_FILE"

# Проверим нужные
: "${DATASET_TRAIN:?Need DATASET_TRAIN in config.sh}"
: "${MODEL:?Need MODEL in config.sh}"
: "${BATCH_SIZE:?Need BATCH_SIZE in config.sh}"
: "${LRS_LIST:?Need LRS_LIST in config.sh}"

DEVICE=${DEVICE:-cpu}
AUTO_DEVICE=${AUTO_DEVICE:-false}
DATALOADER=${DATALOADER:-default}
RESULTS_DIR="src/scripts/exp1/exp_results/setup${SETUP_ID}"

# Путь к питоновскому скрипту
PY_SCRIPT="src/scripts/exp1/python_scripts/val_loss_distribution.py"

# Цикл по всем lr
for LR in $(echo "$LRS_LIST" | tr ',' ' '); do
  WEIGHTS_FILE="${RESULTS_DIR}/sgd_weights_lr${LR}.pt"
  OUT_DIR="${RESULTS_DIR}/val_analysis_lr${LR}"

  if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "[warn] Weights file not found: $WEIGHTS_FILE (skipping)"
    continue
  fi

  echo "[info] setup${SETUP_ID} | lr=${LR}"
  echo "  WEIGHTS_FILE = $WEIGHTS_FILE"
  echo "  OUT_DIR      = $OUT_DIR"

  python3 "$PY_SCRIPT" \
    --weights_file "$WEIGHTS_FILE" \
    --out_dir "$OUT_DIR" \
    --dataset_train "$DATASET_TRAIN" \
    --model "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --data_loader "$DATALOADER" \
    --device "$DEVICE" \
    $( [ "$AUTO_DEVICE" = "true" ] && echo "--auto_device" )
done
