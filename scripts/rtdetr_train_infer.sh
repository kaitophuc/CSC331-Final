#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m src.rtdetr_pipeline pipeline \
  --data "${RTDETR_DATA:-coco8.yaml}" \
  --source "${RTDETR_SOURCE:-https://ultralytics.com/images/bus.jpg}" \
  --model "${RTDETR_MODEL:-rtdetr-l.pt}" \
  --epochs "${RTDETR_EPOCHS:-80}" \
  --imgsz "${RTDETR_IMGSZ:-960}" \
  --batch "${RTDETR_BATCH:-4}" \
  --device "${RTDETR_DEVICE:-cpu}" \
  --optimizer "${RTDETR_OPTIMIZER:-AdamW}" \
  --lr0 "${RTDETR_LR0:-1e-4}" \
  --weight-decay "${RTDETR_WEIGHT_DECAY:-5e-4}" \
  --patience "${RTDETR_PATIENCE:-20}" \
  --cos-lr \
  --amp \
  --train-project "$ROOT_DIR/runs/rtdetr/train" \
  --infer-project "$ROOT_DIR/runs/rtdetr/infer" \
  --train-name "${RTDETR_TRAIN_NAME:-baseline}" \
  --infer-name "${RTDETR_INFER_NAME:-baseline}" \
  --conf "${RTDETR_CONF:-0.25}" \
  --exist-ok \
  "$@"

