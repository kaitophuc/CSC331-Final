#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m src.yolo_pipeline train \
  --data "${YOLO_DATA:-coco8.yaml}" \
  --model "${YOLO_MODEL:-yolo11n.pt}" \
  --epochs "${YOLO_EPOCHS:-50}" \
  --imgsz "${YOLO_IMGSZ:-640}" \
  --batch "${YOLO_BATCH:-16}" \
  --device "${YOLO_DEVICE:-cpu}" \
  --project "$ROOT_DIR/runs/train" \
  --name "${YOLO_TRAIN_NAME:-baseline}" \
  --exist-ok \
  "$@"
