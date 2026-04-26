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
TRT_WEIGHTS ?= runs/detect/runs/train/bdd3_y26m_1280_e15_ft_b4_nomosaic_clean_lr5e4/weights/best.pt
TRT_ENGINE ?=
TRT_SOURCE ?= bus.jpg
TRT_IMGSZ ?= 1280
TRT_BATCH ?= 1
TRT_DEVICE ?= 0
TRT_CONF ?= 0.25
TRT_INFER_NAME ?= yolo26m_trt
TRT_WORKSPACE ?=
TRT_HALF ?= --half
TRT_INSPECT ?= --inspect
TRT_OVERWRITE ?=

.PHONY: train infer pipeline rtdetr-train rtdetr-infer rtdetr-pipeline trt-export trt-infer trt-pipeline

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

trt-export:
	$(PYTHON) infer.py export-trt \
		--weights $(TRT_WEIGHTS) \
		$(if $(TRT_ENGINE),--engine $(TRT_ENGINE),) \
		--imgsz $(TRT_IMGSZ) \
		--batch $(TRT_BATCH) \
		--device $(TRT_DEVICE) \
		$(TRT_HALF) \
		$(if $(TRT_WORKSPACE),--workspace $(TRT_WORKSPACE),) \
		$(TRT_INSPECT) \
		$(TRT_OVERWRITE)

trt-infer:
	$(PYTHON) infer.py infer-trt \
		--engine $(if $(TRT_ENGINE),$(TRT_ENGINE),$(TRT_WEIGHTS:.pt=.engine)) \
		--source $(TRT_SOURCE) \
		--imgsz $(TRT_IMGSZ) \
		--conf $(TRT_CONF) \
		--device $(TRT_DEVICE) \
		--project runs/tensorrt/infer \
		--name $(TRT_INFER_NAME) \
		--exist-ok

trt-pipeline:
	$(PYTHON) infer.py pipeline-trt \
		--weights $(TRT_WEIGHTS) \
		$(if $(TRT_ENGINE),--engine $(TRT_ENGINE),) \
		--source $(TRT_SOURCE) \
		--imgsz $(TRT_IMGSZ) \
		--batch $(TRT_BATCH) \
		--conf $(TRT_CONF) \
		--device $(TRT_DEVICE) \
		--project runs/tensorrt/infer \
		--name $(TRT_INFER_NAME) \
		$(TRT_HALF) \
		$(if $(TRT_WORKSPACE),--workspace $(TRT_WORKSPACE),) \
		$(TRT_INSPECT) \
		$(TRT_OVERWRITE) \
		--exist-ok
