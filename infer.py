from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    ROOT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "train"
    / "bdd3_y26m_1280_e15_ft_b4_nomosaic_clean_lr5e4"
    / "weights"
    / "best.pt"
)
DEFAULT_INFER_PROJECT = ROOT_DIR / "runs" / "tensorrt" / "infer"
DEFAULT_SOURCE = ROOT_DIR / "bus.jpg"
DEFAULT_IMGSZ = 1280
DEFAULT_BATCH = 1
DEFAULT_CONF = 0.25
DEFAULT_DEVICE = "0"


@dataclass
class ExportConfig:
    weights: Path
    engine: Path | None
    imgsz: int
    batch: int
    device: str
    half: bool
    workspace: int | None
    inspect: bool
    overwrite: bool


@dataclass
class InferConfig:
    engine: Path
    source: str
    imgsz: int
    conf: float
    device: str
    project: Path
    name: str
    exist_ok: bool


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "TensorRT export/inference requires a CUDA-visible NVIDIA GPU. "
            "This session reports torch.cuda.is_available() == False."
        )


def require_tensorrt() -> None:
    if importlib.util.find_spec("tensorrt") is None:
        raise SystemExit(
            "TensorRT Python package is not installed in this environment. "
            "Install a platform-compatible tensorrt package on the target GPU machine."
        )


def default_engine_path(weights: Path) -> Path:
    return weights.with_suffix(".engine")


def check_imgsz_stride(model: YOLO, imgsz: int) -> int:
    stride_value = 32
    stride = getattr(getattr(model, "model", None), "stride", None)
    if stride is not None:
        if hasattr(stride, "max"):
            stride_value = int(stride.max().item())
        else:
            stride_value = int(max(stride))
    if imgsz % stride_value != 0:
        raise SystemExit(
            f"imgsz={imgsz} must be divisible by model stride={stride_value} "
            "for a static TensorRT graph."
        )
    return stride_value


def inspect_onnx(onnx_path: Path) -> dict[str, Any] | None:
    if importlib.util.find_spec("onnx") is None:
        print("[inspect] onnx is not installed; skipping ONNX graph inspection.")
        return None
    if not onnx_path.exists():
        print(f"[inspect] ONNX file not found; skipping inspection: {onnx_path}")
        return None

    import onnx

    graph = onnx.load(str(onnx_path)).graph
    op_counts: dict[str, int] = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    activation_ops = {
        "Relu": op_counts.get("Relu", 0),
        "Sigmoid": op_counts.get("Sigmoid", 0),
        "Mul": op_counts.get("Mul", 0),
        "Clip": op_counts.get("Clip", 0),
        "Swish": op_counts.get("Swish", 0),
        "SiLU": op_counts.get("SiLU", 0),
    }
    summary = {
        "onnx": str(onnx_path),
        "nodes": len(graph.node),
        "conv": op_counts.get("Conv", 0),
        "batch_norm": op_counts.get("BatchNormalization", 0),
        "activations": activation_ops,
    }
    print("[inspect] ONNX node summary:")
    print(json.dumps(summary, indent=2))
    if summary["batch_norm"] == 0:
        print("[inspect] BatchNorm nodes are absent; Conv+BN fusion is visible in ONNX.")
    else:
        print(
            "[inspect] BatchNorm nodes remain in ONNX. Review export logs before trusting fusion."
        )
    print(
        "[inspect] TensorRT may still fuse Conv+bias+activation at build time; "
        "confirm final low-level fusion with TensorRT build/profiling logs."
    )
    return summary


def write_export_metadata(
    *,
    weights: Path,
    engine: Path,
    onnx: Path,
    imgsz: int,
    batch: int,
    device: str,
    half: bool,
    stride: int,
    inspect_summary: dict[str, Any] | None,
) -> Path:
    metadata = {
        "weights": str(weights),
        "engine": str(engine),
        "onnx": str(onnx),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "precision": "FP16" if half else "FP32",
        "static_input_shape": [batch, 3, imgsz, imgsz],
        "stride": stride,
        "dynamic": False,
        "nms": False,
        "inspect": inspect_summary,
    }
    metadata_path = engine.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def export_trt(cfg: ExportConfig) -> Path:
    weights = cfg.weights.resolve()
    engine_target = (cfg.engine or default_engine_path(weights)).resolve()
    require_file(weights, "weights")
    require_cuda()
    require_tensorrt()

    if engine_target.exists() and not cfg.overwrite:
        print(f"[export] engine already exists: {engine_target}")
        if cfg.inspect:
            inspect_onnx(engine_target.with_suffix(".onnx"))
        return engine_target

    model = YOLO(str(weights))
    stride = check_imgsz_stride(model, cfg.imgsz)
    print(f"[export] weights: {weights}")
    print(f"[export] static input shape: ({cfg.batch}, 3, {cfg.imgsz}, {cfg.imgsz})")
    print(f"[export] precision: {'FP16' if cfg.half else 'FP32'}")

    exported = Path(
        model.export(
            format="engine",
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            dynamic=False,
            simplify=True,
            half=cfg.half,
            device=cfg.device,
            workspace=cfg.workspace,
            nms=False,
        )
    ).resolve()

    if not exported.exists():
        raise SystemExit(f"TensorRT export did not create an engine file: {exported}")

    engine_target.parent.mkdir(parents=True, exist_ok=True)
    if exported != engine_target:
        if engine_target.exists():
            engine_target.unlink()
        shutil.move(str(exported), str(engine_target))

    onnx_path = weights.with_suffix(".onnx")
    inspect_summary = inspect_onnx(onnx_path) if cfg.inspect else None
    metadata_path = write_export_metadata(
        weights=weights,
        engine=engine_target,
        onnx=onnx_path,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        half=cfg.half,
        stride=stride,
        inspect_summary=inspect_summary,
    )

    print(f"[export] engine: {engine_target}")
    print(f"[export] metadata: {metadata_path}")
    return engine_target


def run_trt_infer(cfg: InferConfig) -> None:
    engine = cfg.engine.resolve()
    require_file(engine, "engine")
    require_cuda()
    require_tensorrt()

    model = YOLO(str(engine))
    results = model.predict(
        source=cfg.source,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        device=cfg.device,
        project=str(cfg.project),
        name=cfg.name,
        save=True,
        exist_ok=cfg.exist_ok,
        stream=True,
    )
    for _ in results:
        pass
    print(f"[infer] predictions saved under: {cfg.project / cfg.name}")


def add_common_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workspace", type=int, default=None, help="TensorRT workspace in GB")
    parser.add_argument("--inspect", action="store_true", help="Inspect exported ONNX graph node counts")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild engine if it already exists")


def add_common_infer_args(
    parser: argparse.ArgumentParser, *, include_imgsz: bool = True, include_device: bool = True
) -> None:
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    if include_imgsz:
        parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    if include_device:
        parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--project", type=Path, default=DEFAULT_INFER_PROJECT)
    parser.add_argument("--name", default="yolo26m_trt")
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Static TensorRT export/inference CLI for the trained YOLO26m checkpoint"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export-trt", help="Export YOLO .pt weights to TensorRT .engine")
    add_common_export_args(export)

    infer = subparsers.add_parser("infer-trt", help="Run inference from a TensorRT .engine")
    infer.add_argument("--engine", type=Path, default=default_engine_path(DEFAULT_WEIGHTS))
    add_common_infer_args(infer)

    pipe = subparsers.add_parser("pipeline-trt", help="Export if needed, then run TensorRT inference")
    add_common_export_args(pipe)
    add_common_infer_args(pipe, include_imgsz=False, include_device=False)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "export-trt":
        export_trt(
            ExportConfig(
                weights=args.weights,
                engine=args.engine,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                half=args.half,
                workspace=args.workspace,
                inspect=args.inspect,
                overwrite=args.overwrite,
            )
        )
        return

    if args.command == "infer-trt":
        run_trt_infer(
            InferConfig(
                engine=args.engine,
                source=args.source,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                project=args.project,
                name=args.name,
                exist_ok=args.exist_ok,
            )
        )
        return

    engine = export_trt(
        ExportConfig(
            weights=args.weights,
            engine=args.engine,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
            workspace=args.workspace,
            inspect=args.inspect,
            overwrite=args.overwrite,
        )
    )
    run_trt_infer(
        InferConfig(
            engine=engine,
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
        )
    )


if __name__ == "__main__":
    main()
