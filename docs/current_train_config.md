# Current Training Config (Local)

This is the active continuation config for YOLO26m on cleaned labels.

## Run Name
- `bdd3_y26m_1280_e15_ft_b4_nomosaic_clean_lr5e4`

## Settings
- `model`: `runs/detect/runs/train/bdd3_y26m_1280_e15_ft_b4/weights/best.pt`
- `data`: `data/yolo_3cls/bdd100k_3cls.yaml`
- `device`: `0`
- `batch`: `4`
- `workers`: `8`
- `optimizer`: `AdamW`
- `lr0`: `0.0005`
- `imgsz`: `1280`
- `epochs`: `15`
- `patience`: `10`
- `cos_lr`: `true`
- `close_mosaic`: `15` (disable mosaic from epoch 1; set equal to epochs)
- `mosaic`: `0.0` (hard disable)
- `amp`: `true`
- `seed`: `0`
- `val`: `true`
- `project`: `runs/train`
- `exist_ok`: `false`

## Command
```bash
venv/bin/python -m src.yolo_pipeline train \
  --model runs/detect/runs/train/bdd3_y26m_1280_e15_ft_b4/weights/best.pt \
  --data data/yolo_3cls/bdd100k_3cls.yaml \
  --device 0 \
  --batch 4 \
  --workers 8 \
  --optimizer AdamW \
  --lr0 0.0005 \
  --imgsz 1280 \
  --epochs 15 \
  --patience 10 \
  --cos-lr \
  --close-mosaic 15 \
  --mosaic 0.0 \
  --amp \
  --seed 0 \
  --val \
  --project runs/train \
  --name bdd3_y26m_1280_e15_ft_b4_nomosaic_clean_lr5e4 \
  --no-exist-ok
```
