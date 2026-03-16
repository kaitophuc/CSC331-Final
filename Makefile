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
RTDETR_DATA ?= coco8.yaml
RTDETR_SOURCE ?= https://ultralytics.com/images/bus.jpg
RTDETR_MODEL ?= rtdetr-l.pt
RTDETR_EPOCHS ?= 80
RTDETR_IMGSZ ?= 960
RTDETR_BATCH ?= 4
RTDETR_DEVICE ?= cpu
RTDETR_CONF ?= 0.25
RTDETR_TRAIN_NAME ?= baseline
RTDETR_INFER_NAME ?= baseline
RTDETR_PATIENCE ?= 20
RTDETR_OPTIMIZER ?= AdamW
RTDETR_LR0 ?= 0.0001
RTDETR_WEIGHT_DECAY ?= 0.0005

.PHONY: train infer pipeline rtdetr-train rtdetr-infer rtdetr-pipeline

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

rtdetr-train:
	$(PYTHON) -m src.rtdetr_pipeline train \
		--data $(RTDETR_DATA) \
		--model $(RTDETR_MODEL) \
		--epochs $(RTDETR_EPOCHS) \
		--imgsz $(RTDETR_IMGSZ) \
		--batch $(RTDETR_BATCH) \
		--device $(RTDETR_DEVICE) \
		--project runs/rtdetr/train \
		--name $(RTDETR_TRAIN_NAME) \
		--optimizer $(RTDETR_OPTIMIZER) \
		--lr0 $(RTDETR_LR0) \
		--weight-decay $(RTDETR_WEIGHT_DECAY) \
		--patience $(RTDETR_PATIENCE) \
		--cos-lr \
		--amp \
		--exist-ok

rtdetr-infer:
	$(PYTHON) -m src.rtdetr_pipeline infer \
		--weights runs/rtdetr/train/$(RTDETR_TRAIN_NAME)/weights/best.pt \
		--source $(RTDETR_SOURCE) \
		--imgsz $(RTDETR_IMGSZ) \
		--conf $(RTDETR_CONF) \
		--device $(RTDETR_DEVICE) \
		--project runs/rtdetr/infer \
		--name $(RTDETR_INFER_NAME) \
		--exist-ok

rtdetr-pipeline:
	$(PYTHON) -m src.rtdetr_pipeline pipeline \
		--data $(RTDETR_DATA) \
		--source $(RTDETR_SOURCE) \
		--model $(RTDETR_MODEL) \
		--epochs $(RTDETR_EPOCHS) \
		--imgsz $(RTDETR_IMGSZ) \
		--batch $(RTDETR_BATCH) \
		--device $(RTDETR_DEVICE) \
		--optimizer $(RTDETR_OPTIMIZER) \
		--lr0 $(RTDETR_LR0) \
		--weight-decay $(RTDETR_WEIGHT_DECAY) \
		--patience $(RTDETR_PATIENCE) \
		--cos-lr \
		--amp \
		--train-project runs/rtdetr/train \
		--infer-project runs/rtdetr/infer \
		--train-name $(RTDETR_TRAIN_NAME) \
		--infer-name $(RTDETR_INFER_NAME) \
		--conf $(RTDETR_CONF) \
		--exist-ok
