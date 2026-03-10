#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m src.yolo_pipeline infer \
  --weights "${YOLO_WEIGHTS:-$ROOT_DIR/runs/train/baseline/weights/best.pt}" \
  --source "${YOLO_SOURCE:-https://ultralytics.com/images/bus.jpg}" \
  --imgsz "${YOLO_IMGSZ:-640}" \
  --conf "${YOLO_CONF:-0.25}" \
  --device "${YOLO_DEVICE:-cpu}" \
  --project "$ROOT_DIR/runs/infer" \
  --name "${YOLO_INFER_NAME:-baseline}" \
  --exist-ok \
  "$@"
