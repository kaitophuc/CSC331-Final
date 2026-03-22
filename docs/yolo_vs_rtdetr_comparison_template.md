# YOLO vs RT-DETR Comparison Template

Use this template to compare both tracks on the same validation split.

## Experiment Metadata

- Date:
- Dataset YAML:
- Split used for evaluation:
- Hardware (GPU/CPU/RAM):
- Random seed:

## Run IDs

- YOLO run name:
- YOLO checkpoint:
- RT-DETR run name:
- RT-DETR checkpoint:

## Training Config Snapshot

| Field | YOLO | RT-DETR |
|---|---|---|
| model |  |  |
| epochs |  |  |
| imgsz |  |  |
| batch |  |  |
| optimizer |  |  |
| lr0 |  |  |
| weight_decay |  |  |
| patience |  |  |
| amp |  |  |

## Validation Metrics

| Metric | YOLO | RT-DETR | Better |
|---|---:|---:|---|
| mAP50-95 |  |  |  |
| mAP50 |  |  |  |
| Precision |  |  |  |
| Recall |  |  |  |

## Per-Class AP (Val)

| Class | YOLO AP | RT-DETR AP | Better |
|---|---:|---:|---|
| traffic_sign |  |  |  |
| pedestrian |  |  |  |
| vehicle |  |  |  |

## Inference Performance

| Metric | YOLO | RT-DETR | Notes |
|---|---:|---:|---|
| latency per image (ms) |  |  |  |
| throughput (img/s) |  |  |  |
| device |  |  |  |

## Stability Checks

- Run repeated with same config at least twice: yes/no
- Metric variance acceptable: yes/no
- Any NaN/divergence observed: yes/no

## Qualitative Review

- Example images reviewed:
- Common YOLO errors:
- Common RT-DETR errors:
- Class-specific failure patterns:

## Decision

- Selected model family:
- Selected checkpoint:
- Reason for decision:
- Follow-up actions:

