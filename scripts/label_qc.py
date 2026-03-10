#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SPLITS: Sequence[str] = ("train", "val", "test")
CLASS_NAMES = {0: "traffic_sign", 1: "pedestrian", 2: "vehicle"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Normalized area bins (w*h with values in [0,1]).
AREA_BINS = (
    ("tiny(<1e-4)", 0.0, 1e-4),
    ("small(1e-4-1e-3)", 1e-4, 1e-3),
    ("medium(1e-3-1e-2)", 1e-3, 1e-2),
    ("large(>=1e-2)", 1e-2, float("inf")),
)

# Aspect ratio bins on max(w/h, h/w), so always >= 1.0
ASPECT_BINS = (
    ("1-2", 1.0, 2.0),
    ("2-3", 2.0, 3.0),
    ("3-5", 3.0, 5.0),
    ("5-10", 5.0, 10.0),
    (">=10", 10.0, float("inf")),
)


@dataclass
class SplitStats:
    image_count: int = 0
    label_count: int = 0
    paired_count: int = 0
    missing_label_for_image: int = 0
    missing_image_for_label: int = 0

    empty_label_files: int = 0
    total_boxes: int = 0
    boxes_per_class: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})

    invalid_line_format: int = 0
    invalid_class_id: int = 0
    invalid_number_parse: int = 0
    out_of_range_coords: int = 0
    non_positive_box: int = 0

    tiny_box_count: int = 0
    extreme_aspect_count: int = 0

    duplicate_exact_extra: int = 0
    duplicate_near_pairs: int = 0
    duplicate_near_boxes: int = 0
    near_dup_skipped_dense_files: int = 0

    area_hist: Dict[str, int] = field(default_factory=lambda: {label: 0 for label, _, _ in AREA_BINS})
    aspect_hist: Dict[str, int] = field(default_factory=lambda: {label: 0 for label, _, _ in ASPECT_BINS})


@dataclass(frozen=True)
class Box:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        if self.w <= 0 or self.h <= 0:
            return float("inf")
        ratio = self.w / self.h
        return ratio if ratio >= 1.0 else 1.0 / ratio

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        x1 = self.xc - self.w / 2.0
        y1 = self.yc - self.h / 2.0
        x2 = self.xc + self.w / 2.0
        y2 = self.yc + self.h / 2.0
        return x1, y1, x2, y2

    def exact_key(self) -> Tuple[int, str, str, str, str]:
        return (
            self.class_id,
            f"{self.xc:.6f}",
            f"{self.yc:.6f}",
            f"{self.w:.6f}",
            f"{self.h:.6f}",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC report for YOLO labels (BDD100K 3-class dataset).")
    p.add_argument("--dataset-dir", type=Path, default=Path("data/yolo_3cls"))
    p.add_argument("--report-path", type=Path, default=Path("docs/bdd100k_3class_label_qc_report.md"))
    p.add_argument("--suspicious-dir", type=Path, default=Path("docs/label_qc"))
    p.add_argument("--near-dup-iou", type=float, default=0.98, help="IoU threshold for near-duplicate boxes.")
    p.add_argument(
        "--near-dup-max-boxes",
        type=int,
        default=200,
        help="Skip near-duplicate pairwise IoU when a single label file has more than this many valid boxes.",
    )
    p.add_argument(
        "--tiny-area-thresh",
        type=float,
        default=1e-4,
        help="Mark suspicious if normalized area (w*h) is below this threshold.",
    )
    p.add_argument(
        "--extreme-aspect-thresh",
        type=float,
        default=10.0,
        help="Mark suspicious if max(w/h, h/w) is above this threshold.",
    )
    p.add_argument(
        "--max-listed-files",
        type=int,
        default=30,
        help="Max suspicious files listed in markdown per split.",
    )
    return p.parse_args()


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def bucket_count(value: float, bins: Iterable[Tuple[str, float, float]]) -> str:
    for label, low, high in bins:
        if low <= value < high:
            return label
    return list(bins)[-1][0]


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def parse_label_line(line: str) -> Tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except Exception:
        return None
    return class_id, xc, yc, w, h


def analyze_split(
    split: str,
    dataset_dir: Path,
    near_dup_iou: float,
    near_dup_max_boxes: int,
    tiny_area_thresh: float,
    extreme_aspect_thresh: float,
) -> Tuple[SplitStats, Dict[str, set[str]]]:
    stats = SplitStats()
    issues_by_file: Dict[str, set[str]] = defaultdict(set)

    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    image_files = [p for p in image_dir.iterdir() if is_image_file(p)] if image_dir.exists() else []
    label_files = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    stats.image_count = len(image_stems)
    stats.label_count = len(label_stems)
    stats.paired_count = len(image_stems & label_stems)

    missing_label = sorted(image_stems - label_stems)
    missing_image = sorted(label_stems - image_stems)
    stats.missing_label_for_image = len(missing_label)
    stats.missing_image_for_label = len(missing_image)

    for stem in missing_label:
        issues_by_file[f"{split}/{stem}.jpg"].add("missing_label_for_image")
    for stem in missing_image:
        issues_by_file[f"{split}/{stem}.txt"].add("missing_image_for_label")

    for label_path in label_files:
        key = f"{split}/{label_path.name}"
        if label_path.stem not in image_stems:
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()
        non_empty_lines = [ln for ln in lines if ln.strip()]
        if not non_empty_lines:
            stats.empty_label_files += 1
            continue

        valid_boxes: List[Box] = []
        for line in non_empty_lines:
            parsed = parse_label_line(line)
            if parsed is None:
                stats.invalid_line_format += 1
                issues_by_file[key].add("invalid_line_format")
                continue
            class_id, xc, yc, w, h = parsed

            if class_id not in CLASS_NAMES:
                stats.invalid_class_id += 1
                issues_by_file[key].add("invalid_class_id")
                continue

            if not all(map(lambda v: v == v and abs(v) != float("inf"), (xc, yc, w, h))):
                stats.invalid_number_parse += 1
                issues_by_file[key].add("invalid_number_parse")
                continue

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                stats.out_of_range_coords += 1
                issues_by_file[key].add("out_of_range_coords")
                continue

            if w <= 0.0 or h <= 0.0:
                stats.non_positive_box += 1
                issues_by_file[key].add("non_positive_box")
                continue

            box = Box(class_id=class_id, xc=xc, yc=yc, w=w, h=h)
            valid_boxes.append(box)

            stats.total_boxes += 1
            stats.boxes_per_class[class_id] += 1
            stats.area_hist[bucket_count(box.area, AREA_BINS)] += 1
            stats.aspect_hist[bucket_count(box.aspect, ASPECT_BINS)] += 1

            if box.area < tiny_area_thresh:
                stats.tiny_box_count += 1
                issues_by_file[key].add("tiny_box")
            if box.aspect >= extreme_aspect_thresh:
                stats.extreme_aspect_count += 1
                issues_by_file[key].add("extreme_aspect")

        if not valid_boxes:
            continue

        exact_counter = Counter(b.exact_key() for b in valid_boxes)
        exact_extra = sum(c - 1 for c in exact_counter.values() if c > 1)
        if exact_extra > 0:
            stats.duplicate_exact_extra += exact_extra
            issues_by_file[key].add("duplicate_exact")

        if len(valid_boxes) > near_dup_max_boxes:
            stats.near_dup_skipped_dense_files += 1
            issues_by_file[key].add("near_dup_skipped_dense_file")
            continue

        by_class: Dict[int, List[Tuple[int, Box]]] = defaultdict(list)
        for i, b in enumerate(valid_boxes):
            by_class[b.class_id].append((i, b))

        near_pair_count = 0
        near_idx = set()
        for class_rows in by_class.values():
            n = len(class_rows)
            for i in range(n):
                idx_i, bi = class_rows[i]
                xy_i = bi.to_xyxy()
                for j in range(i + 1, n):
                    idx_j, bj = class_rows[j]
                    if iou_xyxy(xy_i, bj.to_xyxy()) >= near_dup_iou:
                        near_pair_count += 1
                        near_idx.add(idx_i)
                        near_idx.add(idx_j)

        if near_pair_count > 0:
            stats.duplicate_near_pairs += near_pair_count
            stats.duplicate_near_boxes += len(near_idx)
            issues_by_file[key].add("duplicate_near")

    return stats, issues_by_file


def pct(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return 100.0 * n / d


def write_suspicious_csvs(
    suspicious_dir: Path,
    split_issues: Dict[str, Dict[str, set[str]]],
) -> Dict[str, Path]:
    suspicious_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for split, issues_by_file in split_issues.items():
        out_csv = suspicious_dir / f"{split}_suspicious.csv"
        outputs[split] = out_csv
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "reason_count", "reasons"])
            for file_name in sorted(issues_by_file):
                reasons = sorted(issues_by_file[file_name])
                w.writerow([file_name, len(reasons), ";".join(reasons)])
    return outputs


def generate_report(
    report_path: Path,
    dataset_dir: Path,
    split_stats: Dict[str, SplitStats],
    split_issues: Dict[str, Dict[str, set[str]]],
    suspicious_csv_paths: Dict[str, Path],
    max_listed_files: int,
    near_dup_iou: float,
    tiny_area_thresh: float,
    extreme_aspect_thresh: float,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# BDD100K YOLO Label QC Report")
    lines.append("")
    lines.append(f"- generated: {now}")
    lines.append(f"- dataset: `{dataset_dir}`")
    lines.append(f"- near-duplicate IoU threshold: `{near_dup_iou}`")
    lines.append(f"- tiny area threshold: `{tiny_area_thresh}`")
    lines.append(f"- extreme aspect threshold: `{extreme_aspect_thresh}`")
    lines.append("")
    lines.append("## Split Counts")
    lines.append("")
    lines.append("| Split | Images | Labels | Paired | Missing Label/Image | Missing Image/Label | Empty Labels |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split in SPLITS:
        s = split_stats[split]
        lines.append(
            f"| {split} | {s.image_count} | {s.label_count} | {s.paired_count} | "
            f"{s.missing_label_for_image} | {s.missing_image_for_label} | {s.empty_label_files} |"
        )
    lines.append("")
    lines.append("## Box Counts")
    lines.append("")
    lines.append("| Split | Total Boxes | traffic_sign | pedestrian | vehicle |")
    lines.append("|---|---:|---:|---:|---:|")
    for split in SPLITS:
        s = split_stats[split]
        lines.append(
            f"| {split} | {s.total_boxes} | {s.boxes_per_class[0]} | {s.boxes_per_class[1]} | {s.boxes_per_class[2]} |"
        )
    lines.append("")
    lines.append("## Data Quality Totals")
    lines.append("")
    lines.append("| Split | Invalid Format | Invalid Class | Invalid Number | Out-of-Range | Non-Positive | Tiny Boxes | Extreme Aspect | Duplicate Exact(extra) | Duplicate Near(pairs) | Duplicate Near(boxes) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split in SPLITS:
        s = split_stats[split]
        lines.append(
            f"| {split} | {s.invalid_line_format} | {s.invalid_class_id} | {s.invalid_number_parse} | "
            f"{s.out_of_range_coords} | {s.non_positive_box} | {s.tiny_box_count} | {s.extreme_aspect_count} | "
            f"{s.duplicate_exact_extra} | {s.duplicate_near_pairs} | {s.duplicate_near_boxes} |"
        )
    lines.append("")
    lines.append("## Ratios")
    lines.append("")
    lines.append("| Split | Empty Label % | Tiny Box % | Extreme Aspect % | Dup Exact % (extra/boxes) | Dup Near % (boxes/boxes) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in SPLITS:
        s = split_stats[split]
        lines.append(
            f"| {split} | {pct(s.empty_label_files, s.paired_count):.2f}% | {pct(s.tiny_box_count, s.total_boxes):.3f}% | "
            f"{pct(s.extreme_aspect_count, s.total_boxes):.3f}% | {pct(s.duplicate_exact_extra, s.total_boxes):.3f}% | "
            f"{pct(s.duplicate_near_boxes, s.total_boxes):.3f}% |"
        )
    lines.append("")

    for split in SPLITS:
        s = split_stats[split]
        lines.append(f"## Histograms ({split})")
        lines.append("")
        lines.append("### Area Histogram")
        lines.append("")
        lines.append("| Bin | Count |")
        lines.append("|---|---:|")
        for label, _, _ in AREA_BINS:
            lines.append(f"| {label} | {s.area_hist[label]} |")
        lines.append("")
        lines.append("### Aspect Histogram")
        lines.append("")
        lines.append("| Bin | Count |")
        lines.append("|---|---:|")
        for label, _, _ in ASPECT_BINS:
            lines.append(f"| {label} | {s.aspect_hist[label]} |")
        lines.append("")

    lines.append("## Suspicious Files")
    lines.append("")
    for split in SPLITS:
        issues = split_issues[split]
        lines.append(f"### {split}")
        lines.append("")
        lines.append(f"- suspicious files: {len(issues)}")
        lines.append(f"- csv: `{suspicious_csv_paths[split]}`")
        lines.append("")
        top_items = sorted(issues.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:max_listed_files]
        if not top_items:
            lines.append("No suspicious files detected.")
            lines.append("")
            continue
        lines.append("| File | Reason Count | Reasons |")
        lines.append("|---|---:|---|")
        for file_name, reasons in top_items:
            reason_text = ", ".join(sorted(reasons))
            lines.append(f"| `{file_name}` | {len(reasons)} | {reason_text} |")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir

    split_stats: Dict[str, SplitStats] = {}
    split_issues: Dict[str, Dict[str, set[str]]] = {}

    for split in SPLITS:
        stats, issues = analyze_split(
            split=split,
            dataset_dir=dataset_dir,
            near_dup_iou=args.near_dup_iou,
            near_dup_max_boxes=args.near_dup_max_boxes,
            tiny_area_thresh=args.tiny_area_thresh,
            extreme_aspect_thresh=args.extreme_aspect_thresh,
        )
        split_stats[split] = stats
        split_issues[split] = issues

    suspicious_csv_paths = write_suspicious_csvs(args.suspicious_dir, split_issues)
    generate_report(
        report_path=args.report_path,
        dataset_dir=dataset_dir,
        split_stats=split_stats,
        split_issues=split_issues,
        suspicious_csv_paths=suspicious_csv_paths,
        max_listed_files=args.max_listed_files,
        near_dup_iou=args.near_dup_iou,
        tiny_area_thresh=args.tiny_area_thresh,
        extreme_aspect_thresh=args.extreme_aspect_thresh,
    )

    print(f"QC report: {args.report_path}")
    for split in SPLITS:
        s = split_stats[split]
        print(
            f"[{split}] images={s.image_count} labels={s.label_count} paired={s.paired_count} "
            f"boxes={s.total_boxes} empty={s.empty_label_files} "
            f"issues={len(split_issues[split])} suspicious_csv={suspicious_csv_paths[split]}"
        )


if __name__ == "__main__":
    main()
