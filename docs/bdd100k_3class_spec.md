# BDD100K 3-Class Detection Spec

Status: Draft for approval before JSON-to-YOLO conversion.

## 1) Scope

Train a YOLO detector on BDD100K for exactly 3 classes:

- `0: traffic_sign`
- `1: pedestrian`
- `2: vehicle`

Only 2D object detection boxes are in scope.

## 2) Source Data Layout

Expected local structure:

```
data/
  images/
    train/ val/ test/
  labels/
    train/ val/ test/
```

Annotation format: BDD JSON per image, using `frames[].objects[]`.

## 3) Class Mapping (Approved)

Keep and map these source categories:

- `traffic sign` -> `0 (traffic_sign)`
- `person` -> `1 (pedestrian)`
- `car` -> `2 (vehicle)`
- `truck` -> `2 (vehicle)`
- `bus` -> `2 (vehicle)`
- `train` -> `2 (vehicle)`
- `motorcycle` -> `2 (vehicle)`
- `bicycle` -> `2 (vehicle)`

Explicitly excluded:

- `rider`
- `trailer`
- `other vehicle`

## 4) Annotation Inclusion Rules

A source object is converted only if all are true:

- Category is in the keep list above.
- Object has `box2d` with numeric `x1, y1, x2, y2`.
- Clipped box has positive width and height.

Objects are ignored if:

- Category is not in keep list.
- Object is polygon-only (`poly2d`) or has no `box2d`.
- Box becomes invalid after clipping.

## 5) Bounding Box Rules

Before YOLO normalization:

- Clip `x1, x2` to `[0, image_width]`.
- Clip `y1, y2` to `[0, image_height]`.
- Compute `w = x2 - x1`, `h = y2 - y1`.
- Drop if `w <= 0` or `h <= 0`.

YOLO output format per line:

`<class_id> <x_center> <y_center> <width> <height>`

All coordinates normalized to `[0, 1]`.

## 6) Split Policy

Use existing splits as-is:

- Train: `data/images/train` + labels from `data/labels/train`
- Val: `data/images/val` + labels from `data/labels/val`
- Test: `data/images/test` + labels from `data/labels/test`

No cross-split movement.
No tuning on test.

## 7) Empty-Label Policy

If an image has no remaining objects after filtering, keep it as a negative sample with an empty label file.

## 8) Quality Checks Required at Conversion Time

- Image count matches label JSON count per split.
- Output `.txt` count matches image count per split.
- Report per-class box totals after mapping.
- Report dropped-object counts by reason:
  - category filtered
  - missing `box2d`
  - invalid/zero box after clipping
  - missing image

## 9) Deliverables (Next Step, Not Executed Yet)

- Converted YOLO labels for train/val/test.
- Dataset YAML with:
  - names: `["traffic_sign", "pedestrian", "vehicle"]`
  - split paths
- Conversion report with counts and drop statistics.

## 10) Change Control

Any change to class mapping (especially rider/trailer/other vehicle) requires updating this spec before rerunning conversion.

