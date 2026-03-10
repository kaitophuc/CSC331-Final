# BDD100K YOLO Label QC Report

- generated: 2026-03-01 15:22:10
- dataset: `data/yolo_3cls`
- near-duplicate IoU threshold: `0.98`
- tiny area threshold: `0.0001`
- extreme aspect threshold: `10.0`

## Split Counts

| Split | Images | Labels | Paired | Missing Label/Image | Missing Image/Label | Empty Labels |
|---|---:|---:|---:|---:|---:|---:|
| train | 70000 | 70000 | 70000 | 0 | 0 | 15 |
| val | 10000 | 10000 | 10000 | 0 | 0 | 4 |
| test | 20000 | 20000 | 20000 | 0 | 0 | 7 |

## Box Counts

| Split | Total Boxes | traffic_sign | pedestrian | vehicle |
|---|---:|---:|---:|---:|
| train | 1087038 | 239893 | 91405 | 755740 |
| val | 156533 | 34908 | 13262 | 108363 |
| test | 310673 | 68992 | 24641 | 217040 |

## Data Quality Totals

| Split | Invalid Format | Invalid Class | Invalid Number | Out-of-Range | Non-Positive | Tiny Boxes | Extreme Aspect | Duplicate Exact(extra) | Duplicate Near(pairs) | Duplicate Near(boxes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0 | 0 | 0 | 0 | 0 | 16515 | 1517 | 1 | 2 | 4 |
| val | 0 | 0 | 0 | 0 | 0 | 2269 | 184 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 4915 | 421 | 1 | 1 | 2 |

## Ratios

| Split | Empty Label % | Tiny Box % | Extreme Aspect % | Dup Exact % (extra/boxes) | Dup Near % (boxes/boxes) |
|---|---:|---:|---:|---:|---:|
| train | 0.02% | 1.519% | 0.140% | 0.000% | 0.000% |
| val | 0.04% | 1.450% | 0.118% | 0.000% | 0.000% |
| test | 0.04% | 1.582% | 0.136% | 0.000% | 0.001% |

## Histograms (train)

### Area Histogram

| Bin | Count |
|---|---:|
| tiny(<1e-4) | 16515 |
| small(1e-4-1e-3) | 498769 |
| medium(1e-3-1e-2) | 411190 |
| large(>=1e-2) | 160564 |

### Aspect Histogram

| Bin | Count |
|---|---:|
| 1-2 | 769810 |
| 2-3 | 186524 |
| 3-5 | 94962 |
| 5-10 | 34225 |
| >=10 | 1517 |

## Histograms (val)

### Area Histogram

| Bin | Count |
|---|---:|
| tiny(<1e-4) | 2269 |
| small(1e-4-1e-3) | 72326 |
| medium(1e-3-1e-2) | 59166 |
| large(>=1e-2) | 22772 |

### Aspect Histogram

| Bin | Count |
|---|---:|
| 1-2 | 110885 |
| 2-3 | 26796 |
| 3-5 | 13799 |
| 5-10 | 4869 |
| >=10 | 184 |

## Histograms (test)

### Area Histogram

| Bin | Count |
|---|---:|
| tiny(<1e-4) | 4915 |
| small(1e-4-1e-3) | 142394 |
| medium(1e-3-1e-2) | 117808 |
| large(>=1e-2) | 45556 |

### Aspect Histogram

| Bin | Count |
|---|---:|
| 1-2 | 221113 |
| 2-3 | 53465 |
| 3-5 | 26472 |
| 5-10 | 9202 |
| >=10 | 421 |

## Suspicious Files

### train

- suspicious files: 10307
- csv: `docs/label_qc/train_suspicious.csv`

| File | Reason Count | Reasons |
|---|---:|---|
| `train/0066b72f-974f6883.txt` | 2 | extreme_aspect, tiny_box |
| `train/00f89335-2ef7949d.txt` | 2 | extreme_aspect, tiny_box |
| `train/02346f91-87969c22.txt` | 2 | extreme_aspect, tiny_box |
| `train/029407d6-ca53347d.txt` | 2 | extreme_aspect, tiny_box |
| `train/045d32b5-31a5a2dd.txt` | 2 | extreme_aspect, tiny_box |
| `train/04edbec0-95b33e2a.txt` | 2 | extreme_aspect, tiny_box |
| `train/051ce827-ab278cb8.txt` | 2 | extreme_aspect, tiny_box |
| `train/05266828-33538181.txt` | 2 | extreme_aspect, tiny_box |
| `train/054ce029-b78f0d6e.txt` | 2 | extreme_aspect, tiny_box |
| `train/063eb66e-f58f45b7.txt` | 2 | extreme_aspect, tiny_box |
| `train/08780f20-babe607d.txt` | 2 | extreme_aspect, tiny_box |
| `train/0906211b-4614f770.txt` | 2 | extreme_aspect, tiny_box |
| `train/091d129f-793957ea.txt` | 2 | extreme_aspect, tiny_box |
| `train/09ac21d6-3c3afe65.txt` | 2 | extreme_aspect, tiny_box |
| `train/09c287b9-7c1b1570.txt` | 2 | extreme_aspect, tiny_box |
| `train/0ac6f942-2641cf47.txt` | 2 | extreme_aspect, tiny_box |
| `train/0b396e6f-e75b8338.txt` | 2 | extreme_aspect, tiny_box |
| `train/0ca8467b-6dc2be8a.txt` | 2 | extreme_aspect, tiny_box |
| `train/0ccebe89-e9669633.txt` | 2 | extreme_aspect, tiny_box |
| `train/0ce05c8b-98636a99.txt` | 2 | extreme_aspect, tiny_box |
| `train/0d538703-97a3ecfe.txt` | 2 | extreme_aspect, tiny_box |
| `train/0deb01f1-0ddb5a16.txt` | 2 | extreme_aspect, tiny_box |
| `train/0e9e8f3e-55cd30d7.txt` | 2 | extreme_aspect, tiny_box |
| `train/0f026efa-e39b024a.txt` | 2 | extreme_aspect, tiny_box |
| `train/0f3c486a-75f0f60e.txt` | 2 | extreme_aspect, tiny_box |
| `train/10f71853-4da52f52.txt` | 2 | extreme_aspect, tiny_box |
| `train/115d9e12-fd31cd1b.txt` | 2 | extreme_aspect, tiny_box |
| `train/13041e31-f3a90283.txt` | 2 | extreme_aspect, tiny_box |
| `train/13f87f0e-451dc422.txt` | 2 | extreme_aspect, tiny_box |
| `train/140f97cb-ccb83acf.txt` | 2 | extreme_aspect, tiny_box |

### val

- suspicious files: 1453
- csv: `docs/label_qc/val_suspicious.csv`

| File | Reason Count | Reasons |
|---|---:|---|
| `val/b2c23864-810e4e1d.txt` | 2 | extreme_aspect, tiny_box |
| `val/b2de938e-84eab379.txt` | 2 | extreme_aspect, tiny_box |
| `val/b33ea6cb-8ef8b9c4.txt` | 2 | extreme_aspect, tiny_box |
| `val/b4c733a8-46d78670.txt` | 2 | extreme_aspect, tiny_box |
| `val/b4d9b889-6662ff4a.txt` | 2 | extreme_aspect, tiny_box |
| `val/b5e247fe-c9599f82.txt` | 2 | extreme_aspect, tiny_box |
| `val/b7a9b166-a7b6abec.txt` | 2 | extreme_aspect, tiny_box |
| `val/b7bfad97-89e3ef28.txt` | 2 | extreme_aspect, tiny_box |
| `val/b8a795ee-c1ce7ac0.txt` | 2 | extreme_aspect, tiny_box |
| `val/b8acc534-d5b64200.txt` | 2 | extreme_aspect, tiny_box |
| `val/b93c2c35-9a03812d.txt` | 2 | extreme_aspect, tiny_box |
| `val/bb16329e-7ddd640e.txt` | 2 | extreme_aspect, tiny_box |
| `val/bb5cc516-eb91e8c8.txt` | 2 | extreme_aspect, tiny_box |
| `val/bb71ba86-3204e97e.txt` | 2 | extreme_aspect, tiny_box |
| `val/bc05c126-d00aeb9c.txt` | 2 | extreme_aspect, tiny_box |
| `val/bc7bc1e5-b3223500.txt` | 2 | extreme_aspect, tiny_box |
| `val/bcc84a33-2b38c69f.txt` | 2 | extreme_aspect, tiny_box |
| `val/be26ac57-35dabb9c.txt` | 2 | extreme_aspect, tiny_box |
| `val/be651ed5-02659bee.txt` | 2 | extreme_aspect, tiny_box |
| `val/bf42f468-f2b3301f.txt` | 2 | extreme_aspect, tiny_box |
| `val/bf4906e0-73ed04de.txt` | 2 | extreme_aspect, tiny_box |
| `val/c0082590-c1533b81.txt` | 2 | extreme_aspect, tiny_box |
| `val/c12ea027-2919c581.txt` | 2 | extreme_aspect, tiny_box |
| `val/c239f9d5-92e82ab6.txt` | 2 | extreme_aspect, tiny_box |
| `val/c242b4b8-6757e7b6.txt` | 2 | extreme_aspect, tiny_box |
| `val/c29d444d-06b31782.txt` | 2 | extreme_aspect, tiny_box |
| `val/c422771c-1619481d.txt` | 2 | extreme_aspect, tiny_box |
| `val/c564c92b-dec2638b.txt` | 2 | extreme_aspect, tiny_box |
| `val/c5b2506d-1c31cee3.txt` | 2 | extreme_aspect, tiny_box |
| `val/c6648ad5-91058550.txt` | 2 | extreme_aspect, tiny_box |

### test

- suspicious files: 2957
- csv: `docs/label_qc/test_suspicious.csv`

| File | Reason Count | Reasons |
|---|---:|---|
| `test/caeb0fbb-410c10ca.txt` | 2 | extreme_aspect, tiny_box |
| `test/cb031b03-cc89642d.txt` | 2 | extreme_aspect, tiny_box |
| `test/cbe712c2-e12e21b5.txt` | 2 | extreme_aspect, tiny_box |
| `test/cbf622f5-06c371dd.txt` | 2 | extreme_aspect, tiny_box |
| `test/cc0e38f5-72db5b25.txt` | 2 | extreme_aspect, tiny_box |
| `test/cc892afa-0b55cf6e.txt` | 2 | extreme_aspect, tiny_box |
| `test/cd0c8cf0-d78ca88f.txt` | 2 | extreme_aspect, tiny_box |
| `test/ce8a6ad4-078b291e.txt` | 2 | extreme_aspect, tiny_box |
| `test/cf3e60d2-d8fa8bfb.txt` | 2 | extreme_aspect, tiny_box |
| `test/cfaac5ea-e8a5deb9.txt` | 2 | extreme_aspect, tiny_box |
| `test/cfc1934b-b876008c.txt` | 2 | extreme_aspect, tiny_box |
| `test/d15925d8-941565e4.txt` | 2 | extreme_aspect, tiny_box |
| `test/d1b19d09-053351ca.txt` | 2 | extreme_aspect, tiny_box |
| `test/d1ba04ca-3e0e41ab.txt` | 2 | extreme_aspect, tiny_box |
| `test/d1ea2d9c-d935c212.txt` | 2 | extreme_aspect, tiny_box |
| `test/d27f30e7-6e2f4c72.txt` | 2 | extreme_aspect, tiny_box |
| `test/d2bf2973-b37bd264.txt` | 2 | extreme_aspect, tiny_box |
| `test/d2f7022b-fb6cd551.txt` | 2 | extreme_aspect, tiny_box |
| `test/d381a921-e44d5b0b.txt` | 2 | extreme_aspect, tiny_box |
| `test/d468e05b-5f126d92.txt` | 2 | extreme_aspect, tiny_box |
| `test/d4726ee9-c77d66d7.txt` | 2 | extreme_aspect, tiny_box |
| `test/d4c08810-6e774772.txt` | 2 | extreme_aspect, tiny_box |
| `test/d65a28ae-442c6d80.txt` | 2 | extreme_aspect, tiny_box |
| `test/d6fbc02b-32af9055.txt` | 2 | extreme_aspect, tiny_box |
| `test/d758bb34-9fca69ff.txt` | 2 | extreme_aspect, tiny_box |
| `test/d76129bc-6f2895de.txt` | 2 | extreme_aspect, tiny_box |
| `test/d7873426-9ec6fd25.txt` | 2 | extreme_aspect, tiny_box |
| `test/d795d9d8-d6101cf4.txt` | 2 | extreme_aspect, tiny_box |
| `test/d7ded4ff-949cfb69.txt` | 2 | extreme_aspect, tiny_box |
| `test/d870856a-6503afd4.txt` | 2 | extreme_aspect, tiny_box |

