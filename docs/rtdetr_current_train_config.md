# Current RT-DETR Training Config (Local)

Use this file to track the active RT-DETR continuation config.

## Run Name

- `bdd3_rtdetr_l_baseline`

## Settings (Baseline Defaults)

- `model`: `rtdetr-l.pt`
- `data`: `data/yolo_3cls/bdd100k_3cls.yaml`
- `device`: `0`
- `batch`: `4`
- `workers`: `8`
- `optimizer`: `AdamW`
- `lr0`: `0.0001`
- `weight_decay`: `0.0005`
- `imgsz`: `960`
- `epochs`: `80`
- `patience`: `20`
- `cos_lr`: `true`
- `amp`: `true`
- `seed`: `0`
- `val`: `true`
- `project`: `runs/rtdetr/train`
- `exist_ok`: `false` (recommended for final experiments)

## Smoke Run (Preflight)

```bash
venv/bin/python -m src.rtdetr_pipeline train \
  --model rtdetr-l.pt \
  --data data/yolo_3cls/bdd100k_3cls.yaml \
  --device 0 \
  --batch 2 \
  --imgsz 640 \
  --epochs 1 \
  --optimizer AdamW \
  --lr0 0.0001 \
  --weight-decay 0.0005 \
  --patience 20 \
  --cos-lr \
  --amp \
  --val \
  --project runs/rtdetr/train \
  --name smoke \
  --exist-ok
```

## Baseline Fine-Tune Command

```bash
venv/bin/python -m src.rtdetr_pipeline train \
  --model rtdetr-l.pt \
  --data data/yolo_3cls/bdd100k_3cls.yaml \
  --device 0 \
  --batch 4 \
  --workers 8 \
  --optimizer AdamW \
  --lr0 0.0001 \
  --weight-decay 0.0005 \
  --imgsz 960 \
  --epochs 80 \
  --patience 20 \
  --cos-lr \
  --amp \
  --seed 0 \
  --val \
  --project runs/rtdetr/train \
  --name bdd3_rtdetr_l_baseline \
  --no-exist-ok
```

