import os
import subprocess
from pathlib import Path

import modal

app = modal.App("yolo-bdd100k-train")
DATA_YAML = "/vol/data/yolo_3cls/bdd100k_3cls_modal.yaml"
DEFAULT_MODEL = "yolo11m.pt"
YOLO_CONFIG_ROOT = Path("/tmp/yolo_config")

data_vol = modal.Volume.from_name("bdd100k-data", create_if_missing=True, version=2)
runs_vol = modal.Volume.from_name("bdd100k-runs", create_if_missing=True, version=2)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            ".git",
            "venv",
            "data",
            "runs",
            "__pycache__",
            "logs_*.txt",
            "*.log",
        ],
    )
)

@app.function(
    image=image,
    gpu="L40S",              # or "A10", "A100", "H100", etc.
    cpu=8.0,
    memory=32768,
    ephemeral_disk=524288,
    timeout=60 * 60 * 20,    # up to 24h
    startup_timeout=60 * 30,
    volumes={"/vol/data": data_vol, "/vol/runs": runs_vol},
)
def train(
    run_name: str = "bdd3_modal_m1280_smoke",
    epochs: int = 1,
    imgsz: int = 1280,
    batch: int = -1,
    model: str = DEFAULT_MODEL,
):
    # Temporary: train directly from mounted volume to skip startup data copy.

    cmd = [
        "python", "-m", "src.yolo_pipeline", "train",
        "--data", DATA_YAML,
        "--model", model,
        "--epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
        "--device", "0",
        "--project", "/vol/runs/train",
        "--name", run_name,
        "--exist-ok",
    ]
    # Ensure Ultralytics can always write settings/cache in a writable location.
    (YOLO_CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "HOME": "/tmp",
        "YOLO_CONFIG_DIR": str(YOLO_CONFIG_ROOT),
    }
    subprocess.run(cmd, cwd="/root/project", env=env, check=True)
    runs_vol.commit()  # persist weights to Volume
    return f"/train/{run_name}/weights/best.pt"

@app.local_entrypoint()
def main(
    run_name: str = "bdd3_modal_m1280_smoke",
    epochs: int = 1,
    imgsz: int = 1280,
    batch: int = -1,
    model: str = DEFAULT_MODEL,
):
    best = train.remote(run_name, epochs, imgsz, batch, model)
    print("Best weight in volume:", best)
