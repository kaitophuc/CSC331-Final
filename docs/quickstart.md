# YOLO Baseline Quickstart

Implementation choice: **Ultralytics YOLO** (fastest path to training/inference baseline).

## 1) Create/activate environment

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Prepare dataset

Add your YOLO dataset under `data/` and create `data/dataset.yaml`.
See `data/README.md`.

## 3) Run training + inference in one command

```
make pipeline
```

This default command uses `coco8.yaml` and a sample image URL.
Use your own dataset/source by overriding variables:

```
make pipeline DATA=data/dataset.yaml SOURCE=data/infer DEVICE=0
```

Alternative shell wrapper (same defaults):

```
./scripts/train_infer.sh
```

## 4) Separate commands

```
make train DATA=data/dataset.yaml DEVICE=0
make infer SOURCE=data/infer DEVICE=0
```

## 5) TensorRT YOLO26m inference

The dedicated TensorRT CLI is `infer.py`. It uses Ultralytics export so the
model is converted to a static ONNX graph, Conv+BatchNorm is fused into Conv
with bias before export, and TensorRT builds the final `.engine`.

```
venv/bin/python infer.py export-trt --inspect
venv/bin/python infer.py infer-trt --source bus.jpg
```

Equivalent Makefile shortcuts:

```
make trt-export
make trt-infer
make trt-pipeline
```

Defaults target:

```
runs/detect/runs/train/bdd3_y26m_1280_e15_ft_b4_nomosaic_clean_lr5e4/weights/best.pt
```

TensorRT export/inference requires an NVIDIA GPU-visible session and a
platform-compatible TensorRT install. Expected extras are `onnx`, `onnxslim`,
`onnxruntime-gpu`, and `tensorrt`. The current local shell has CUDA-built
PyTorch but no visible GPU and no TensorRT/ONNX packages, so `export-trt` should
stop early here with an environment message; build the `.engine` on the target
GPU machine where it will be used.
