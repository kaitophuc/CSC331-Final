#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

SPLITS = ("train", "val", "test")


@dataclass
class CleanStats:
    train_files_seen: int = 0
    train_files_written: int = 0
    train_files_changed: int = 0
    train_files_empty_after_clean: int = 0

    input_boxes: int = 0
    output_boxes: int = 0

    dropped_invalid_format: int = 0
    dropped_invalid_number: int = 0
    dropped_invalid_range: int = 0
    dropped_non_positive: int = 0
    dropped_tiny: int = 0
    dropped_extreme_aspect: int = 0
    dropped_exact_duplicate: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a cleaned YOLO dataset copy by filtering noisy TRAIN labels only. "
            "Source dataset is never modified."
        )
    )
    p.add_argument("--source-dir", type=Path, default=Path("data/yolo_3cls"))
    p.add_argument("--output-dir", type=Path, default=Path("data/yolo_3cls_cleaned"))
    p.add_argument(
        "--image-link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to place images in output dataset.",
    )
    p.add_argument(
        "--label-link-mode",
        choices=("symlink", "copy"),
        default="copy",
        help="How to place untouched val/test labels in output dataset.",
    )
    p.add_argument(
        "--tiny-area-thresh",
        type=float,
        default=5e-5,
        help="Drop train boxes where normalized area (w*h) is below this threshold.",
    )
    p.add_argument(
        "--extreme-aspect-thresh",
        type=float,
        default=12.0,
        help="Drop train boxes where max(w/h, h/w) exceeds this threshold.",
    )
    p.add_argument(
        "--dedupe-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop exact duplicate train boxes (same class + same 6-decimal coords).",
    )
    p.add_argument(
        "--keep-empty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep empty train label files after cleaning (recommended).",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete output-dir before writing if it already exists.",
    )
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Analyze and report only; do not write output files.",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/label_qc/train_label_cleaning_report.md"),
    )
    p.add_argument(
        "--changes-csv",
        type=Path,
        default=Path("docs/label_qc/train_label_cleaning_changes.csv"),
    )
    return p.parse_args()


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {path}\n"
                "Pass --overwrite to replace it, or pick a new --output-dir."
            )
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_dir(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)

    if mode == "symlink":
        rel_target = Path(os.path.relpath(src, dst.parent))
        dst.symlink_to(rel_target, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def parse_line(line: str) -> Tuple[int, float, float, float, float] | None:
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


def format_line(class_id: int, xc: float, yc: float, w: float, h: float) -> str:
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def is_finite(values: Iterable[float]) -> bool:
    for v in values:
        if v != v or v == float("inf") or v == float("-inf"):
            return False
    return True


def clean_train_labels(
    src_train_labels: Path,
    dst_train_labels: Path,
    stats: CleanStats,
    tiny_area_thresh: float,
    extreme_aspect_thresh: float,
    dedupe_exact: bool,
    keep_empty: bool,
    dry_run: bool,
) -> List[Tuple[str, int, int, int]]:
    changes: List[Tuple[str, int, int, int]] = []
    label_files = sorted(src_train_labels.glob("*.txt"))
    stats.train_files_seen = len(label_files)

    if not dry_run:
        dst_train_labels.mkdir(parents=True, exist_ok=True)

    for label_path in label_files:
        in_lines = label_path.read_text(encoding="utf-8").splitlines()
        out_lines: List[str] = []
        seen_exact = set()
        input_boxes_file = 0

        for raw in in_lines:
            raw = raw.strip()
            if not raw:
                continue
            input_boxes_file += 1
            stats.input_boxes += 1

            parsed = parse_line(raw)
            if parsed is None:
                # Could be format or parse issue; treat as invalid format for simplicity.
                stats.dropped_invalid_format += 1
                continue
            class_id, xc, yc, w, h = parsed

            if not is_finite((xc, yc, w, h)):
                stats.dropped_invalid_number += 1
                continue

            if class_id not in (0, 1, 2):
                stats.dropped_invalid_range += 1
                continue

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                stats.dropped_invalid_range += 1
                continue

            if w <= 0.0 or h <= 0.0:
                stats.dropped_non_positive += 1
                continue

            area = w * h
            if area < tiny_area_thresh:
                stats.dropped_tiny += 1
                continue

            ratio = w / h if w >= h else h / w
            if ratio > extreme_aspect_thresh:
                stats.dropped_extreme_aspect += 1
                continue

            key = (class_id, f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}")
            if dedupe_exact and key in seen_exact:
                stats.dropped_exact_duplicate += 1
                continue
            seen_exact.add(key)
            out_lines.append(format_line(class_id, xc, yc, w, h))
            stats.output_boxes += 1

        kept = len(out_lines)
        dropped = input_boxes_file - kept
        if dropped > 0:
            stats.train_files_changed += 1
            changes.append((label_path.name, input_boxes_file, kept, dropped))

        if kept == 0:
            stats.train_files_empty_after_clean += 1
            if not keep_empty:
                continue

        if not dry_run:
            out_path = dst_train_labels / label_path.name
            out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            stats.train_files_written += 1

    return changes


def copy_or_link_non_train_labels(source_dir: Path, out_dir: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        return
    for split in ("val", "test"):
        src = source_dir / "labels" / split
        dst = out_dir / "labels" / split
        dst.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy_dir(src, dst, mode)


def write_dataset_yaml(source_dir: Path, out_dir: Path, dry_run: bool) -> None:
    src_yaml = source_dir / "bdd100k_3cls.yaml"
    dst_yaml = out_dir / "bdd100k_3cls.yaml"
    if dry_run:
        return
    if src_yaml.exists():
        text = src_yaml.read_text(encoding="utf-8")
        # Keep relative path style used by current project.
        rel_path = out_dir.as_posix()
        text = text.replace("path: data/yolo_3cls", f"path: {rel_path}")
    else:
        text = (
            f"path: {out_dir.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names: [traffic_sign, pedestrian, vehicle]\n"
        )
    dst_yaml.write_text(text, encoding="utf-8")


def write_changes_csv(path: Path, changes: List[Tuple[str, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "boxes_before", "boxes_after", "boxes_dropped"])
        for row in sorted(changes, key=lambda r: (-r[3], r[0])):
            w.writerow(row)


def write_report(
    report_path: Path,
    source_dir: Path,
    out_dir: Path,
    stats: CleanStats,
    tiny_area_thresh: float,
    extreme_aspect_thresh: float,
    dedupe_exact: bool,
    keep_empty: bool,
    dry_run: bool,
    changes_csv: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    now = os.popen("date '+%Y-%m-%d %H:%M:%S'").read().strip()
    lines: List[str] = []
    lines.append("# Train Label Cleaning Report")
    lines.append("")
    lines.append(f"- generated: {now}")
    lines.append(f"- source dataset: `{source_dir}`")
    lines.append(f"- output dataset: `{out_dir}`")
    lines.append(f"- dry_run: `{dry_run}`")
    lines.append("")
    lines.append("## Cleaning Rules (train labels only)")
    lines.append("")
    lines.append(f"- tiny area threshold: `{tiny_area_thresh}`")
    lines.append(f"- extreme aspect threshold: `{extreme_aspect_thresh}`")
    lines.append(f"- dedupe exact: `{dedupe_exact}`")
    lines.append(f"- keep empty files: `{keep_empty}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- train files seen: `{stats.train_files_seen}`")
    lines.append(f"- train files written: `{stats.train_files_written}`")
    lines.append(f"- train files changed: `{stats.train_files_changed}`")
    lines.append(f"- train files empty after clean: `{stats.train_files_empty_after_clean}`")
    lines.append(f"- input boxes: `{stats.input_boxes}`")
    lines.append(f"- output boxes: `{stats.output_boxes}`")
    lines.append("")
    lines.append("## Drop Reasons")
    lines.append("")
    lines.append(f"- invalid format: `{stats.dropped_invalid_format}`")
    lines.append(f"- invalid number: `{stats.dropped_invalid_number}`")
    lines.append(f"- invalid range/class: `{stats.dropped_invalid_range}`")
    lines.append(f"- non-positive width/height: `{stats.dropped_non_positive}`")
    lines.append(f"- tiny boxes: `{stats.dropped_tiny}`")
    lines.append(f"- extreme aspect: `{stats.dropped_extreme_aspect}`")
    lines.append(f"- exact duplicates: `{stats.dropped_exact_duplicate}`")
    lines.append("")
    lines.append(f"- changes csv: `{changes_csv}`")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir: Path = args.source_dir
    out_dir: Path = args.output_dir

    src_images = source_dir / "images"
    src_labels = source_dir / "labels"
    src_train_labels = src_labels / "train"

    if not src_images.exists() or not src_labels.exists() or not src_train_labels.exists():
        raise FileNotFoundError(f"Invalid source dataset layout at: {source_dir}")

    stats = CleanStats()

    if not args.dry_run:
        ensure_output_root(out_dir, overwrite=args.overwrite)
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)

        # Images: link/copy all splits from source.
        for split in SPLITS:
            link_or_copy_dir(src_images / split, out_dir / "images" / split, args.image_link_mode)

    # Labels: clean train split into new output.
    changes = clean_train_labels(
        src_train_labels=src_train_labels,
        dst_train_labels=out_dir / "labels" / "train",
        stats=stats,
        tiny_area_thresh=args.tiny_area_thresh,
        extreme_aspect_thresh=args.extreme_aspect_thresh,
        dedupe_exact=args.dedupe_exact,
        keep_empty=args.keep_empty,
        dry_run=args.dry_run,
    )

    # Labels: keep val/test untouched via copy/symlink.
    copy_or_link_non_train_labels(source_dir, out_dir, args.label_link_mode, args.dry_run)
    write_dataset_yaml(source_dir, out_dir, args.dry_run)

    # Reports
    write_changes_csv(args.changes_csv, changes)
    write_report(
        report_path=args.report_path,
        source_dir=source_dir,
        out_dir=out_dir,
        stats=stats,
        tiny_area_thresh=args.tiny_area_thresh,
        extreme_aspect_thresh=args.extreme_aspect_thresh,
        dedupe_exact=args.dedupe_exact,
        keep_empty=args.keep_empty,
        dry_run=args.dry_run,
        changes_csv=args.changes_csv,
    )

    print("Train label cleaning complete.")
    print(f"Source dataset: {source_dir}")
    print(f"Output dataset: {out_dir}")
    print(
        f"Train files seen={stats.train_files_seen}, changed={stats.train_files_changed}, "
        f"input_boxes={stats.input_boxes}, output_boxes={stats.output_boxes}"
    )
    print(
        "Dropped: "
        f"invalid_format={stats.dropped_invalid_format}, "
        f"invalid_number={stats.dropped_invalid_number}, "
        f"invalid_range={stats.dropped_invalid_range}, "
        f"non_positive={stats.dropped_non_positive}, "
        f"tiny={stats.dropped_tiny}, "
        f"extreme_aspect={stats.dropped_extreme_aspect}, "
        f"duplicate_exact={stats.dropped_exact_duplicate}"
    )
    print(f"Changes CSV: {args.changes_csv}")
    print(f"Report: {args.report_path}")
    if args.dry_run:
        print("Dry-run mode: no dataset files were written.")


if __name__ == "__main__":
    main()
