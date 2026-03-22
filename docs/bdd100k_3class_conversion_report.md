# BDD100K 3-Class Conversion Report

Generated: `2026-02-27T15:56:39`

Output dataset: `data/yolo_3cls`
Dataset YAML: `data/yolo_3cls/bdd100k_3cls.yaml`

## Split Counts

| Split | Images | JSON Labels | Output TXT Labels |
|---|---:|---:|---:|
| train | 70000 | 70000 | 70000 |
| val | 10000 | 10000 | 10000 |
| test | 20000 | 20000 | 20000 |
| total | 100000 | 100000 | 100000 |

## Per-Class Box Totals (After Mapping)

- `0 traffic_sign`: 343793
- `1 pedestrian`: 129308
- `2 vehicle`: 1081143

## Dropped-Object Totals by Reason

- filtered category: 1223046
- missing box2d: 0
- invalid after clipping: 443
- missing image: 0

## Sanity Check (20 Random Image+Label Pairs)

- sampled labels: 20
- sampled images opened: 20
- invalid lines: 0
- invalid class IDs (outside 0,1,2): 0
- invalid coordinates: 0

Sampled label files:
- `test/d400cf5a-20fe6c30.txt`
- `train/250e7595-b14da270.txt`
- `train/0879ba7f-c8c678db.txt`
- `test/f6862d55-cccc90e9.txt`
- `train/5b9a4827-93212b5c.txt`
- `train/5205c74c-cba68560.txt`
- `train/4acf1374-24d997be.txt`
- `train/2ea16819-1eac86f0.txt`
- `test/f4d3056e-7f6f09f9.txt`
- `train/223046e4-a58ad883.txt`
- `test/e13ffec9-0f52f6a3.txt`
- `test/f631192d-04e585c6.txt`
- `val/b569a737-a960c153.txt`
- `train/1d6acdbb-e99c93e6.txt`
- `val/c43daa45-612283ce.txt`
- `train/8d0dc4cc-5792c505.txt`
- `train/0abdc803-ee7ab185.txt`
- `train/0a0379a5-2a18e95b.txt`
- `train/1f8f289b-31727be1.txt`
- `train/495900ca-847af9ca.txt`
