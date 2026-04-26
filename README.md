# CSC331 Final - BDD100K Object Detection

This repository contains the CSC331 Group 2 final project for training and comparing object detection models on a 3-class BDD100K setup. The project converts BDD100K labels into YOLO format, checks label quality, fine-tunes YOLO and RT-DETR models, runs inference, and records the workflow in a final Jupyter notebook.

Target classes:

- `traffic_sign`
- `pedestrian`
- `vehicle`

The main notebook is [`final-project-group2-1.ipynb`](final-project-group2-1.ipynb). A local working copy is also included as [`final-project-group2-1_local.ipynb`](final-project-group2-1_local.ipynb).

## Project Workflow

1. Prepare the Python environment from [`requirements.txt`](requirements.txt).
2. Download or mount BDD100K data under `data/`.
3. Convert BDD100K annotations into the 3-class YOLO dataset format.
4. Train a YOLO model, including the YOLO26m experiment path.
5. Train an RT-DETR model on the same dataset format.
6. Run inference and compare predictions, metrics, and training behavior.
7. Document commands, outputs, and conclusions in the final notebook.

## Quick Start

Create and activate the environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the default YOLO train-plus-infer pipeline:

```bash
make pipeline
```

Run with a local dataset and inference source:

```bash
make pipeline DATA=data/dataset.yaml SOURCE=data/infer DEVICE=0
```

Run the RT-DETR pipeline:

```bash
make rtdetr-pipeline RTDETR_DATA=data/yolo_3cls/bdd100k_3cls.yaml RTDETR_SOURCE=data/infer RTDETR_DEVICE=0
```

See [`docs/quickstart.md`](docs/quickstart.md) for the YOLO workflow and [`docs/rtdetr_quickstart.md`](docs/rtdetr_quickstart.md) for the RT-DETR workflow.

## Notebook Guide

Open [`final-project-group2-1.ipynb`](final-project-group2-1.ipynb) from the repository root in Jupyter, VS Code, or Google Colab.

The notebook is organized around:

- setup and dependency checks
- BDD100K download/data preparation
- YOLO26m training
- RT-DETR training
- model comparison and final project notes

If running locally, use the project virtual environment as the notebook kernel. If running on Colab or another GPU machine, make sure the dataset paths match the notebook cells and the YAML file used by the training commands.

Useful notebook support docs:

- [`docs/ipynb_workflow_plan.md`](docs/ipynb_workflow_plan.md): planned notebook structure and deliverables
- [`docs/current_train_config.md`](docs/current_train_config.md): current YOLO training configuration
- [`docs/rtdetr_current_train_config.md`](docs/rtdetr_current_train_config.md): current RT-DETR training configuration
- [`docs/yolo_vs_rtdetr_comparison_template.md`](docs/yolo_vs_rtdetr_comparison_template.md): comparison table template

## Repository Structure

```text
.
├── final-project-group2-1.ipynb        # Main final project notebook
├── final-project-group2-1_local.ipynb  # Local notebook copy
├── README.md                           # Project overview and guide
├── requirements.txt                    # Python dependencies
├── Makefile                            # Common training/inference commands
├── infer.py                            # TensorRT export/inference CLI for YOLO weights
├── modal_train.py                      # Modal GPU training entrypoint
├── bdd100k_3cls_modal.yaml             # Dataset YAML for Modal volume paths
├── data/                               # Local datasets and inference inputs, gitignored except docs
├── docs/                               # Setup notes, configs, reports, and workflow docs
├── scripts/                            # Shell wrappers for YOLO, RT-DETR, conversion, and QC
├── src/                                # Python CLIs for YOLO and RT-DETR pipelines
└── runs/                               # Training and inference outputs, gitignored except placeholder
```

## Important Folders and Files

- [`src/yolo_pipeline.py`](src/yolo_pipeline.py): train, infer, and one-command pipeline CLI for Ultralytics YOLO.
- [`src/rtdetr_pipeline.py`](src/rtdetr_pipeline.py) and [`src/rtdetr/pipeline.py`](src/rtdetr/pipeline.py): train, infer, and pipeline CLI for RT-DETR.
- [`scripts/convert_bdd_to_yolo.py`](scripts/convert_bdd_to_yolo.py): converts BDD100K annotations into the approved 3-class YOLO format.
- [`scripts/label_qc.py`](scripts/label_qc.py): checks converted YOLO labels and produces QC outputs.
- [`scripts/clean_train_labels.py`](scripts/clean_train_labels.py): applies documented label cleaning changes.
- [`docs/bdd100k_3class_spec.md`](docs/bdd100k_3class_spec.md): class mapping, conversion rules, split policy, and quality requirements.
- [`docs/bdd100k_3class_conversion_report.md`](docs/bdd100k_3class_conversion_report.md): conversion counts and drop statistics.
- [`docs/bdd100k_3class_label_qc_report.md`](docs/bdd100k_3class_label_qc_report.md): label QC summary and suspicious-file reports.

## Common Commands

YOLO training:

```bash
make train DATA=data/yolo_3cls/bdd100k_3cls.yaml MODEL=yolo26m.pt DEVICE=0 TRAIN_NAME=bdd3_yolo26m
```

YOLO inference:

```bash
make infer SOURCE=data/infer DEVICE=0 TRAIN_NAME=bdd3_yolo26m INFER_NAME=bdd3_yolo26m
```

RT-DETR training:

```bash
make rtdetr-train RTDETR_DATA=data/yolo_3cls/bdd100k_3cls.yaml RTDETR_DEVICE=0 RTDETR_TRAIN_NAME=bdd3_rtdetr_l
```

RT-DETR inference:

```bash
make rtdetr-infer RTDETR_SOURCE=data/infer RTDETR_DEVICE=0 RTDETR_TRAIN_NAME=bdd3_rtdetr_l RTDETR_INFER_NAME=bdd3_rtdetr_l
```

TensorRT export and inference for the trained YOLO checkpoint:

```bash
make trt-pipeline
```

TensorRT requires an NVIDIA GPU-visible environment plus compatible TensorRT, ONNX, and ONNX Runtime GPU packages.

## Data and Outputs

`data/` is intended for local datasets and inference inputs. It is not committed to the repository. See [`data/README.md`](data/README.md) for the expected YOLO dataset layout.

`runs/` is where training checkpoints, validation outputs, and prediction images are written. It is also not committed, so keep any final metrics, screenshots, or checkpoint paths recorded in the notebook and docs.
