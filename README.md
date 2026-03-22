# CSC331 Final - YOLO Baseline

This repository is set up with an Ultralytics YOLO baseline for object detection.

## One-command train + infer

```
make pipeline
```

Default command uses Ultralytics sample dataset (`coco8.yaml`) and sample image source.
For your own data, override:

```
make pipeline DATA=data/dataset.yaml SOURCE=data/infer DEVICE=0
```

## Structure

- `data/` local dataset/inference inputs (gitignored)
- `src/` Python training/inference CLI
- `scripts/` shell wrappers
- `runs/` model outputs (gitignored)
- `docs/` environment and quickstart docs
