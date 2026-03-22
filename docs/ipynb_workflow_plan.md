# Final Project Notebook Plan

This notebook documents all code and results completed so far, in two training stages:

1. Fine-tune `yolo26m`
2. Fine-tune `rtdetr-l` (same config style currently used in this project folder)

## 1) Goal

- Train and compare two object detection models on the same dataset split and evaluation flow.
- Keep training setup reproducible with explicit config + run names.
- Collect key outputs: training logs, best weights, validation metrics, and sample predictions.

## 2) Environment and Setup

- Project root: `CSC331-Final`
- Python env: `venv`
- Dataset YAML: `data/yolo_3cls/bdd100k_3cls.yaml` (or your active YAML)
- Device: GPU if available (`0`), else `cpu`

## 3) Stage A: Fine-tune YOLO26m

### A.1 Config Source

- Base config reference: `docs/current_train_config.md`
- Model: `yolo26m.pt` (or continuation checkpoint from previous run)

### A.2 Training Run

- Run training with the same hyperparameter pattern in your current config.
- Save outputs under `runs/train/<run_name>/`.

### A.3 Outputs to Record in Notebook

- Final command used
- Training/validation metrics table
- Best checkpoint path (`best.pt`)
- Example inference visualizations
- Notes on failure cases or class confusion

## 4) Stage B: Fine-tune RT-DETR-L

### B.1 Config Source

- Config reference: `docs/rtdetr_current_train_config.md`
- Model: `rtdetr-l.pt`
- Keep settings consistent with the config already used in this folder.

### B.2 Training Run

- Run RT-DETR training using the same dataset and comparable training setup.
- Save outputs under `runs/rtdetr/train/<run_name>/`.

### B.3 Outputs to Record in Notebook

- Final command used
- Training/validation metrics table
- Best checkpoint path (`best.pt`)
- Example inference visualizations
- Notes on strengths/weaknesses vs YOLO26m

## 5) Comparison Section (YOLO26m vs RT-DETR-L)

- Compare:
- mAP / precision / recall
- Training time and resource usage
- Qualitative prediction quality on the same images
- Error patterns by class (`car`, `traffic sign`, `traffic light`)

## 6) Reproducibility Checklist

- Dataset YAML path fixed and shown
- Random seed shown
- Model weights path shown
- Exact commands captured
- Output folders and checkpoints linked

## 7) Final Deliverables in This Notebook

- End-to-end code cells for both training pipelines
- Logged outputs and metric snapshots
- Side-by-side model comparison summary
- Short conclusion on which model to carry forward and why
