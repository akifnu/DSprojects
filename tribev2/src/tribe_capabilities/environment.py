from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EnvironmentReport:
    python_version: str
    platform: str
    cuda_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    numpy_version: str | None
    torch_version: str | None
    tribev2_importable: bool
    hf_token_present: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ready_for_inference(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _query_gpu() -> tuple[str | None, float | None]:
    if shutil.which("nvidia-smi") is None:
        return None, None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = result.stdout.strip().splitlines()[0]
        name, memory_mb = [part.strip() for part in line.split(",", maxsplit=1)]
        return name, round(float(memory_mb) / 1024, 1)
    except (subprocess.CalledProcessError, IndexError, ValueError):
        return None, None


def check_environment(
    *,
    min_vram_gb: float = 40.0,
    require_tribev2: bool = False,
) -> EnvironmentReport:
    warnings: list[str] = []
    errors: list[str] = []

    numpy_version = None
    try:
        import numpy as np

        numpy_version = np.__version__
        if numpy_version != "2.2.6":
            warnings.append(
                "TRIBE v2 expects numpy==2.2.6; other versions may break neuralset imports."
            )
    except ImportError:
        warnings.append("NumPy is not installed.")

    torch_version = None
    cuda_available = False
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        warnings.append("PyTorch is not installed.")

    tribev2_importable = False
    try:
        import tribev2  # noqa: F401

        tribev2_importable = True
    except ImportError:
        message = "tribev2 is not installed."
        if require_tribev2:
            errors.append(message)
        else:
            warnings.append(message)

    gpu_name, gpu_vram_gb = _query_gpu()
    if cuda_available and gpu_vram_gb is not None and gpu_vram_gb < min_vram_gb:
        warnings.append(
            f"Detected {gpu_vram_gb:.1f} GB VRAM; full trimodal inference needs "
            f"at least {min_vram_gb:.0f} GB."
        )
    if not cuda_available:
        warnings.append("CUDA is unavailable. GPU inference tests will be skipped.")

    hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not hf_token_present:
        warnings.append(
            "No HuggingFace token found. Text inference requires LLaMA 3.2 access."
        )

    return EnvironmentReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        numpy_version=numpy_version,
        torch_version=torch_version,
        tribev2_importable=tribev2_importable,
        hf_token_present=hf_token_present,
        warnings=warnings,
        errors=errors,
    )
