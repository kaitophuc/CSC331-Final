#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m src.yolo_pipeline pipeline \
  --data "${YOLO_DATA:-coco8.yaml}" \
  --source "${YOLO_SOURCE:-https://ultralytics.com/images/bus.jpg}" \
  --model "${YOLO_MODEL:-yolo11n.pt}" \
  --epochs "${YOLO_EPOCHS:-50}" \
  --imgsz "${YOLO_IMGSZ:-640}" \
  --batch "${YOLO_BATCH:-16}" \
  --device "${YOLO_DEVICE:-cpu}" \
  --train-project "$ROOT_DIR/runs/train" \
  --infer-project "$ROOT_DIR/runs/infer" \
  --train-name "${YOLO_TRAIN_NAME:-baseline}" \
  --infer-name "${YOLO_INFER_NAME:-baseline}" \
  --conf "${YOLO_CONF:-0.25}" \
  --exist-ok \
  "$@"
