# RT-DETR Quickstart (Parallel Track)

This project keeps YOLO unchanged and adds RT-DETR as a separate workflow.

## 1) Create/activate environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Prepare dataset

Use the same YOLO-format dataset YAML used by YOLO runs, for example:

```bash
data/yolo_3cls/bdd100k_3cls.yaml
```

Class order must remain:

1. `traffic_sign`
2. `pedestrian`
3. `vehicle`

## 3) Preflight smoke run (recommended)

```bash
make rtdetr-train RTDETR_DATA=data/yolo_3cls/bdd100k_3cls.yaml RTDETR_EPOCHS=1 RTDETR_IMGSZ=640 RTDETR_BATCH=2 RTDETR_DEVICE=0 RTDETR_TRAIN_NAME=smoke
make rtdetr-infer RTDETR_SOURCE=https://ultralytics.com/images/bus.jpg RTDETR_DEVICE=0 RTDETR_TRAIN_NAME=smoke RTDETR_INFER_NAME=smoke
```

Expected output:

- `runs/rtdetr/train/smoke/weights/best.pt`
- `runs/rtdetr/infer/smoke/*`

## 4) Baseline fine-tune run

```bash
make rtdetr-train RTDETR_DATA=data/yolo_3cls/bdd100k_3cls.yaml RTDETR_DEVICE=0 RTDETR_TRAIN_NAME=bdd3_rtdetr_l_baseline
```

Current defaults:

- `RTDETR_MODEL=rtdetr-l.pt`
- `RTDETR_EPOCHS=80`
- `RTDETR_IMGSZ=960`
- `RTDETR_BATCH=4`
- `RTDETR_OPTIMIZER=AdamW`
- `RTDETR_LR0=1e-4`
- `RTDETR_WEIGHT_DECAY=5e-4`
- `RTDETR_PATIENCE=20`

## 5) One-command train + infer

```bash
make rtdetr-pipeline RTDETR_DATA=data/yolo_3cls/bdd100k_3cls.yaml RTDETR_SOURCE=https://ultralytics.com/images/bus.jpg RTDETR_DEVICE=0 RTDETR_TRAIN_NAME=bdd3_rtdetr_l_baseline RTDETR_INFER_NAME=bdd3_rtdetr_l_baseline
```

## 6) Optional shell wrappers

```bash
./scripts/rtdetr_train.sh --data data/yolo_3cls/bdd100k_3cls.yaml --device 0
./scripts/rtdetr_infer.sh --source https://ultralytics.com/images/bus.jpg --device 0
./scripts/rtdetr_train_infer.sh --data data/yolo_3cls/bdd100k_3cls.yaml --source https://ultralytics.com/images/bus.jpg --device 0
```

