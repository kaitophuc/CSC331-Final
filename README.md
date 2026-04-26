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

## TensorRT inference

The TensorRT path targets the current best YOLO26m checkpoint:

```
make trt-pipeline
```

This exports `runs/detect/runs/train/bdd3_y26m_1280_e15_ft_b4_nomosaic_clean_lr5e4/weights/best.pt`
to a static TensorRT `.engine`, then runs inference with that engine.

TensorRT export must run on an NVIDIA GPU-visible session with compatible
`tensorrt`, `onnx`, `onnxslim`, and `onnxruntime-gpu` packages installed. This
local shell has CUDA-built PyTorch, but currently reports no visible GPU and no
TensorRT/ONNX packages, so export is expected to fail early here with a clear
environment message.

## Structure

- `data/` local dataset/inference inputs (gitignored)
- `src/` Python training/inference CLI
- `scripts/` shell wrappers
- `runs/` model outputs (gitignored)
- `docs/` environment and quickstart docs
