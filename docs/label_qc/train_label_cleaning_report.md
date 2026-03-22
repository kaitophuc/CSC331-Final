# Train Label Cleaning Report

- generated: 2026-03-01 15:40:42
- source dataset: `data/yolo_3cls`
- output dataset: `/tmp/yolo_3cls_clean_build`
- dry_run: `False`

## Cleaning Rules (train labels only)

- tiny area threshold: `5e-05`
- extreme aspect threshold: `12.0`
- dedupe exact: `True`
- keep empty files: `True`

## Summary

- train files seen: `70000`
- train files written: `70000`
- train files changed: `1703`
- train files empty after clean: `15`
- input boxes: `1087038`
- output boxes: `1084961`

## Drop Reasons

- invalid format: `0`
- invalid number: `0`
- invalid range/class: `0`
- non-positive width/height: `0`
- tiny boxes: `1446`
- extreme aspect: `630`
- exact duplicates: `1`

- changes csv: `docs/label_qc/train_label_cleaning_changes.csv`

