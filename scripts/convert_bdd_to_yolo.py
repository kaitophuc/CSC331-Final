#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

SPLITS = ("train", "val", "test")
CLASS_MAP = {
    "traffic sign": 0,
    "person": 1,
    "car": 2,
    "truck": 2,
    "bus": 2,
    "train": 2,
    "motorcycle": 2,
    "bicycle": 2,
}
CLASS_NAMES = ["traffic_sign", "pedestrian", "vehicle"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class SplitStats:
    image_count: int = 0
    json_count: int = 0
    txt_count: int = 0
    boxes_per_class: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})
    dropped_filtered_category: int = 0
    dropped_missing_box2d: int = 0
    dropped_invalid_after_clipping: int = 0
    dropped_missing_image: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BDD100K JSON labels to YOLO 3-class labels.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/yolo_3cls"))
    parser.add_argument("--report-path", type=Path, default=Path("docs/bdd100k_3class_conversion_report.md"))
    parser.add_argument("--image-link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--sanity-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def mirror_images(src_images_root: Path, out_images_root: Path, mode: str) -> None:
    out_images_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = src_images_root / split
        dst = out_images_root / split
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)

        if mode == "symlink":
            rel_target = Path(os.path.relpath(src, out_images_root))
            dst.symlink_to(rel_target, target_is_directory=True)
        else:
            shutil.copytree(src, dst)


def find_image_path(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_objects(bdd_json: dict) -> List[dict]:
    frames = bdd_json.get("frames")
    if not isinstance(frames, list) or not frames:
        return []
    frame0 = frames[0]
    objects = frame0.get("objects", [])
    if isinstance(objects, list):
        return objects
    return []


def clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float]:
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return x1, y1, x2, y2


def to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    box_w = x2 - x1
    box_h = y2 - y1
    xc = x1 + box_w / 2.0
    yc = y1 + box_h / 2.0
    return f"{class_id} {xc / width:.6f} {yc / height:.6f} {box_w / width:.6f} {box_h / height:.6f}"


def convert_split(
    split: str,
    images_root: Path,
    labels_json_root: Path,
    labels_out_root: Path,
) -> SplitStats:
    stats = SplitStats()
    image_dir = images_root / split
    json_dir = labels_json_root / split
    out_dir = labels_out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in image_dir.iterdir() if p.is_file()]
    json_files = sorted(json_dir.glob("*.json"))
    stats.image_count = len(image_files)
    stats.json_count = len(json_files)

    for json_path in json_files:
        stem = json_path.stem
        image_path = find_image_path(image_dir, stem)
        if image_path is None:
            stats.dropped_missing_image += 1
            continue

        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception:
            stats.dropped_missing_image += 1
            continue

        data = load_json(json_path)
        objects = extract_objects(data)
        lines: List[str] = []

        for obj in objects:
            category = obj.get("category")
            if category not in CLASS_MAP:
                stats.dropped_filtered_category += 1
                continue

            box = obj.get("box2d")
            if not isinstance(box, dict):
                stats.dropped_missing_box2d += 1
                continue

            try:
                x1 = float(box["x1"])
                y1 = float(box["y1"])
                x2 = float(box["x2"])
                y2 = float(box["y2"])
            except Exception:
                stats.dropped_missing_box2d += 1
                continue

            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, width, height)
            if x2 <= x1 or y2 <= y1:
                stats.dropped_invalid_after_clipping += 1
                continue

            class_id = CLASS_MAP[category]
            lines.append(to_yolo_line(class_id, x1, y1, x2, y2, width, height))
            stats.boxes_per_class[class_id] += 1

        out_label = out_dir / f"{stem}.txt"
        out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats.txt_count += 1

    return stats


def write_dataset_yaml(output_dir: Path) -> Path:
    yaml_path = output_dir / "bdd100k_3cls.yaml"
    text = (
        "path: data/yolo_3cls\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names: [traffic_sign, pedestrian, vehicle]\n"
    )
    yaml_path.write_text(text, encoding="utf-8")
    return yaml_path


@dataclass
class SanityResult:
    sampled_labels: int
    sampled_images_opened: int
    invalid_lines: int
    invalid_class_ids: int
    invalid_coordinates: int
    sample_files: List[str]


def run_sanity_check(output_dir: Path, samples: int, seed: int) -> SanityResult:
    label_files: List[Path] = []
    for split in SPLITS:
        label_files.extend(sorted((output_dir / "labels" / split).glob("*.txt")))

    random.seed(seed)
    if not label_files:
        return SanityResult(0, 0, 0, 0, 0, [])
    sampled = random.sample(label_files, k=min(samples, len(label_files)))

    invalid_lines = 0
    invalid_class_ids = 0
    invalid_coordinates = 0
    opened_images = 0
    sample_names: List[str] = []

    for label_path in sampled:
        split = label_path.parent.name
        stem = label_path.stem
        sample_names.append(f"{split}/{stem}.txt")
        image_path = find_image_path(output_dir / "images" / split, stem)
        if image_path is None:
            invalid_coordinates += 1
            continue

        with Image.open(image_path) as im:
            width, height = im.size
            opened_images += 1

        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_lines += 1
                continue

            try:
                class_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except Exception:
                invalid_lines += 1
                continue

            if class_id not in (0, 1, 2):
                invalid_class_ids += 1

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                invalid_coordinates += 1
                continue

            x1 = (xc - bw / 2.0) * width
            x2 = (xc + bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            y2 = (yc + bh / 2.0) * height
            if x2 <= x1 or y2 <= y1:
                invalid_coordinates += 1
                continue
            if x1 < -1e-3 or y1 < -1e-3 or x2 > width + 1e-3 or y2 > height + 1e-3:
                invalid_coordinates += 1

    return SanityResult(
        sampled_labels=len(sampled),
        sampled_images_opened=opened_images,
        invalid_lines=invalid_lines,
        invalid_class_ids=invalid_class_ids,
        invalid_coordinates=invalid_coordinates,
        sample_files=sample_names,
    )


def write_report(
    report_path: Path,
    stats: Dict[str, SplitStats],
    sanity: SanityResult,
    output_dir: Path,
    yaml_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")

    total_boxes = {0: 0, 1: 0, 2: 0}
    totals = SplitStats()
    for split_stats in stats.values():
        totals.image_count += split_stats.image_count
        totals.json_count += split_stats.json_count
        totals.txt_count += split_stats.txt_count
        totals.dropped_filtered_category += split_stats.dropped_filtered_category
        totals.dropped_missing_box2d += split_stats.dropped_missing_box2d
        totals.dropped_invalid_after_clipping += split_stats.dropped_invalid_after_clipping
        totals.dropped_missing_image += split_stats.dropped_missing_image
        for cid in (0, 1, 2):
            total_boxes[cid] += split_stats.boxes_per_class[cid]

    lines = [
        "# BDD100K 3-Class Conversion Report",
        "",
        f"Generated: `{now}`",
        "",
        f"Output dataset: `{output_dir}`",
        f"Dataset YAML: `{yaml_path}`",
        "",
        "## Split Counts",
        "",
        "| Split | Images | JSON Labels | Output TXT Labels |",
        "|---|---:|---:|---:|",
    ]

    for split in SPLITS:
        s = stats[split]
        lines.append(f"| {split} | {s.image_count} | {s.json_count} | {s.txt_count} |")
    lines.append(f"| total | {totals.image_count} | {totals.json_count} | {totals.txt_count} |")
    lines.extend(
        [
            "",
            "## Per-Class Box Totals (After Mapping)",
            "",
            f"- `0 traffic_sign`: {total_boxes[0]}",
            f"- `1 pedestrian`: {total_boxes[1]}",
            f"- `2 vehicle`: {total_boxes[2]}",
            "",
            "## Dropped-Object Totals by Reason",
            "",
            f"- filtered category: {totals.dropped_filtered_category}",
            f"- missing box2d: {totals.dropped_missing_box2d}",
            f"- invalid after clipping: {totals.dropped_invalid_after_clipping}",
            f"- missing image: {totals.dropped_missing_image}",
            "",
            "## Sanity Check (20 Random Image+Label Pairs)",
            "",
            f"- sampled labels: {sanity.sampled_labels}",
            f"- sampled images opened: {sanity.sampled_images_opened}",
            f"- invalid lines: {sanity.invalid_lines}",
            f"- invalid class IDs (outside 0,1,2): {sanity.invalid_class_ids}",
            f"- invalid coordinates: {sanity.invalid_coordinates}",
            "",
            "Sampled label files:",
        ]
    )
    for sample in sanity.sample_files:
        lines.append(f"- `{sample}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    images_src = data_dir / "images"
    labels_json_src = data_dir / "labels"
    labels_out = output_dir / "labels"
    images_out = output_dir / "images"

    ensure_clean_dir(labels_out)
    mirror_images(images_src, images_out, args.image_link_mode)

    split_stats: Dict[str, SplitStats] = {}
    for split in SPLITS:
        split_stats[split] = convert_split(split, images_src, labels_json_src, labels_out)

    yaml_path = write_dataset_yaml(output_dir)
    sanity = run_sanity_check(output_dir, args.sanity_samples, args.seed)
    write_report(args.report_path, split_stats, sanity, output_dir, yaml_path)

    print("Conversion complete.")
    print(f"Output labels: {labels_out}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"QC report: {args.report_path}")
    for split in SPLITS:
        s = split_stats[split]
        print(
            f"[{split}] images={s.image_count} json={s.json_count} txt={s.txt_count} "
            f"boxes={{0:{s.boxes_per_class[0]},1:{s.boxes_per_class[1]},2:{s.boxes_per_class[2]}}} "
            f"drops={{filtered:{s.dropped_filtered_category},missing_box2d:{s.dropped_missing_box2d},"
            f"invalid:{s.dropped_invalid_after_clipping},missing_image:{s.dropped_missing_image}}}"
        )
    print(
        "Sanity: "
        f"samples={sanity.sampled_labels}, opened={sanity.sampled_images_opened}, "
        f"invalid_lines={sanity.invalid_lines}, invalid_class_ids={sanity.invalid_class_ids}, "
        f"invalid_coordinates={sanity.invalid_coordinates}"
    )


if __name__ == "__main__":
    main()
