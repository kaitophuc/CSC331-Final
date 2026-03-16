#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m src.rtdetr_pipeline infer \
  --weights "${RTDETR_WEIGHTS:-$ROOT_DIR/runs/rtdetr/train/baseline/weights/best.pt}" \
  --source "${RTDETR_SOURCE:-https://ultralytics.com/images/bus.jpg}" \
  --imgsz "${RTDETR_IMGSZ:-960}" \
  --conf "${RTDETR_CONF:-0.25}" \
  --device "${RTDETR_DEVICE:-cpu}" \
  --project "$ROOT_DIR/runs/rtdetr/infer" \
  --name "${RTDETR_INFER_NAME:-baseline}" \
  --exist-ok \
  "$@"

