PYTHON ?= venv/bin/python
DATA ?= coco8.yaml
SOURCE ?= https://ultralytics.com/images/bus.jpg
MODEL ?= yolo11n.pt
EPOCHS ?= 50
IMGSZ ?= 640
BATCH ?= 16
DEVICE ?= cpu
CONF ?= 0.25
TRAIN_NAME ?= baseline
INFER_NAME ?= baseline
FRACTION ?= 1.0
VAL ?= --val
PATIENCE ?= 100

.PHONY: train infer pipeline

train:
	$(PYTHON) -m src.yolo_pipeline train \
		--data $(DATA) \
		--model $(MODEL) \
		--epochs $(EPOCHS) \
		--imgsz $(IMGSZ) \
		--batch $(BATCH) \
		--device $(DEVICE) \
		--project runs/train \
		--name $(TRAIN_NAME) \
		--patience $(PATIENCE) \
		--exist-ok

infer:
	$(PYTHON) -m src.yolo_pipeline infer \
		--weights runs/train/$(TRAIN_NAME)/weights/best.pt \
		--source $(SOURCE) \
		--imgsz $(IMGSZ) \
		--conf $(CONF) \
		--device $(DEVICE) \
		--project runs/infer \
		--name $(INFER_NAME) \
		--exist-ok

pipeline:
	$(PYTHON) -m src.yolo_pipeline pipeline \
		--data $(DATA) \
		--source $(SOURCE) \
		--model $(MODEL) \
		--epochs $(EPOCHS) \
		--imgsz $(IMGSZ) \
		--batch $(BATCH) \
		--device $(DEVICE) \
		--train-project runs/train \
		--infer-project runs/infer \
		--train-name $(TRAIN_NAME) \
		--infer-name $(INFER_NAME) \
		--conf $(CONF) \
		--exist-ok
