# Environment Baseline

Captured on: `2026-02-27T09:58:05-05:00`

## Python

- Python: `3.12.3`
- pip: `24.0`

## Core ML Packages

- ultralytics: `8.4.18`
- torch: `2.10.0+cu128`
- torchvision: `0.25.0`

## CUDA / GPU

- PyTorch CUDA build: `12.8`
- `torch.cuda.is_available()`: `False`
- `torch.cuda.device_count()`: `0`
- `nvidia-smi`: unavailable (`Failed to initialize NVML: Unknown Error`)

Notes:
- The installed PyTorch build includes CUDA (`+cu128`), but this shell session cannot access an NVIDIA GPU.
- Re-run `nvidia-smi` and a torch CUDA check on the target training machine to confirm runtime GPU availability.

