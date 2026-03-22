from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ultralytics import RTDETR

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PROJECT = ROOT_DIR / "runs" / "rtdetr" / "train"
DEFAULT_INFER_PROJECT = ROOT_DIR / "runs" / "rtdetr" / "infer"


@dataclass
class TrainConfig:
    model: str
    data: str
    epochs: int
    imgsz: int
    batch: int
    device: str
    project: Path
    name: str
    exist_ok: bool
    patience: int = 20
    fraction: float = 1.0
    val: bool = True
    workers: int = 8
    optimizer: str = "AdamW"
    cos_lr: bool = True
    amp: bool = True
    seed: int = 0
    lr0: float = 1e-4
    weight_decay: float = 5e-4


@dataclass
class InferConfig:
    weights: str
    source: str
    imgsz: int
    conf: float
    device: str
    project: Path
    name: str
    exist_ok: bool


def run_train(cfg: TrainConfig) -> Path:
    model = RTDETR(cfg.model)
    model.train(
        data=cfg.data,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=str(cfg.project),
        name=cfg.name,
        exist_ok=cfg.exist_ok,
        patience=cfg.patience,
        fraction=cfg.fraction,
        val=cfg.val,
        workers=cfg.workers,
        optimizer=cfg.optimizer,
        cos_lr=cfg.cos_lr,
        amp=cfg.amp,
        seed=cfg.seed,
        lr0=cfg.lr0,
        weight_decay=cfg.weight_decay,
    )
    if getattr(model, "trainer", None) is not None:
        best = Path(model.trainer.best)
        last = Path(model.trainer.last)
        weights = best if best.exists() else last
    else:
        weights = cfg.project / cfg.name / "weights" / "best.pt"
    print(f"[rtdetr train] weights for inference: {weights}")
    return weights


def run_infer(cfg: InferConfig) -> None:
    model = RTDETR(cfg.weights)
    model.predict(
        source=cfg.source,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        device=cfg.device,
        project=str(cfg.project),
        name=cfg.name,
        save=True,
        exist_ok=cfg.exist_ok,
    )
    print(f"[rtdetr infer] predictions saved under: {cfg.project / cfg.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ultralytics RT-DETR CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train an RT-DETR model")
    train.add_argument("--model", default="rtdetr-l.pt")
    train.add_argument("--data", required=True, help="Path to dataset YAML")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--imgsz", type=int, default=960)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--device", default="cpu", help="cpu, 0, 0,1, etc.")
    train.add_argument("--project", type=Path, default=DEFAULT_TRAIN_PROJECT)
    train.add_argument("--name", default="baseline")
    train.add_argument("--patience", type=int, default=20, help="Early stopping patience in epochs")
    train.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use per epoch (0.0-1.0)",
    )
    train.add_argument(
        "--val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation at end of each epoch",
    )
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--optimizer", default="AdamW")
    train.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--lr0", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=5e-4)
    train.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the same run directory without incrementing names",
    )

    infer = subparsers.add_parser("infer", help="Run inference with RT-DETR")
    infer.add_argument("--weights", required=True, help="Path to model weights")
    infer.add_argument("--source", required=True, help="Image/video/folder source")
    infer.add_argument("--imgsz", type=int, default=960)
    infer.add_argument("--conf", type=float, default=0.25)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--project", type=Path, default=DEFAULT_INFER_PROJECT)
    infer.add_argument("--name", default="baseline")
    infer.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    pipe = subparsers.add_parser(
        "pipeline", help="Train and then run inference in one command"
    )
    pipe.add_argument("--model", default="rtdetr-l.pt")
    pipe.add_argument("--data", required=True, help="Path to dataset YAML")
    pipe.add_argument("--source", required=True, help="Image/video/folder source")
    pipe.add_argument("--epochs", type=int, default=80)
    pipe.add_argument("--imgsz", type=int, default=960)
    pipe.add_argument("--batch", type=int, default=4)
    pipe.add_argument("--device", default="cpu")
    pipe.add_argument("--patience", type=int, default=20)
    pipe.add_argument("--fraction", type=float, default=1.0)
    pipe.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    pipe.add_argument("--workers", type=int, default=8)
    pipe.add_argument("--optimizer", default="AdamW")
    pipe.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    pipe.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    pipe.add_argument("--seed", type=int, default=0)
    pipe.add_argument("--lr0", type=float, default=1e-4)
    pipe.add_argument("--weight-decay", type=float, default=5e-4)
    pipe.add_argument("--train-project", type=Path, default=DEFAULT_TRAIN_PROJECT)
    pipe.add_argument("--infer-project", type=Path, default=DEFAULT_INFER_PROJECT)
    pipe.add_argument("--train-name", default="baseline")
    pipe.add_argument("--infer-name", default="baseline")
    pipe.add_argument("--conf", type=float, default=0.25)
    pipe.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(
            TrainConfig(
                model=args.model,
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project=args.project,
                name=args.name,
                exist_ok=args.exist_ok,
                patience=args.patience,
                fraction=args.fraction,
                val=args.val,
                workers=args.workers,
                optimizer=args.optimizer,
                cos_lr=args.cos_lr,
                amp=args.amp,
                seed=args.seed,
                lr0=args.lr0,
                weight_decay=args.weight_decay,
            )
        )
        return

    if args.command == "infer":
        run_infer(
            InferConfig(
                weights=args.weights,
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

    weights = run_train(
        TrainConfig(
            model=args.model,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.train_project,
            name=args.train_name,
            exist_ok=args.exist_ok,
            patience=args.patience,
            fraction=args.fraction,
            val=args.val,
            workers=args.workers,
            optimizer=args.optimizer,
            cos_lr=args.cos_lr,
            amp=args.amp,
            seed=args.seed,
            lr0=args.lr0,
            weight_decay=args.weight_decay,
        )
    )
    run_infer(
        InferConfig(
            weights=str(weights),
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            project=args.infer_project,
            name=args.infer_name,
            exist_ok=args.exist_ok,
        )
    )


if __name__ == "__main__":
    main()

