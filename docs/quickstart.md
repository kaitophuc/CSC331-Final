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
