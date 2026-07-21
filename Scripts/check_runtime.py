#!/usr/bin/env python3
"""Fail-fast check for the realtime T2 Python/CUDA/model environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "bec6edd7e195bab863132f1e9659cdd26e6eaeae7cfd24a626828de294cf5b3a"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "Model/MACVO_FrontendCov.pth")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python >=3.10 is required, got {platform.python_version()}")

    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("torch", "pypose", "cv2", "yaml", "numpy", "scipy", "timm", "einops"):
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "installed"))

    import torch

    versions["cuda_runtime"] = str(torch.version.cuda)
    versions["cuda_available"] = str(torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable PyTorch installation and NVIDIA GPU are required")

    for module in ("DataLoader", "Module", "Odometry.MACVO", "Utility.TwoStateVIO", "Utility.LiveDashboard"):
        importlib.import_module(module)

    if not args.skip_model:
        model = args.model.expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Missing {model}; run python Scripts/download_models.py")
        current = sha256(model)
        if current != MODEL_SHA256:
            raise RuntimeError(f"Model SHA-256 {current}, expected {MODEL_SHA256}")

    print("Realtime T2 runtime check passed")
    for key, value in versions.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
