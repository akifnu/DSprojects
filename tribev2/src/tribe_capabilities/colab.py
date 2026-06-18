from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tribe_capabilities.environment import check_environment
from tribe_capabilities.inference import apply_torch_compat_patches, clear_cuda_cache


INSTALL_MARKER = Path("/content/.tribev2_colab_ready_v5")
DEFAULT_CHECKPOINT = Path("/content/framing_rct_checkpoint.json")
SCENARIOS_URL = (
    "https://raw.githubusercontent.com/akifnu/DSprojects/main/"
    "tribev2/data/framing_rct/scenarios.json"
)


@dataclass
class GpuReport:
    name: str | None
    vram_gb: float | None
    cuda_available: bool
    warnings: list[str]


def query_gpu() -> GpuReport:
    report = check_environment(min_vram_gb=0.0, require_tribev2=False)
    warnings = list(report.warnings)
    if report.gpu_vram_gb is not None and report.gpu_vram_gb < 35:
        warnings.append(
            f"Detected {report.gpu_vram_gb:.1f} GB VRAM. TRIBE text inference is safest on "
            "A100 (40 GB). If you hit OOM, lower MAX_SCENARIOS and restart runtime."
        )
    if not report.cuda_available:
        warnings.append("CUDA is not available. Set Runtime → Change runtime type → GPU.")
    return GpuReport(
        name=report.gpu_name,
        vram_gb=report.gpu_vram_gb,
        cuda_available=report.cuda_available,
        warnings=warnings,
    )


def ensure_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        return token
    except Exception:
        pass

    from getpass import getpass

    token = getpass("Hugging Face read token (gated LLaMA access): ").strip()
    if not token:
        raise RuntimeError("HF token is required for TRIBE text inference.")
    os.environ["HF_TOKEN"] = token
    return token


def login_huggingface() -> None:
    from huggingface_hub import login

    token = ensure_hf_token()
    login(token=token, add_to_git_credential=False)


def install_colab_dependencies(requirements_url: str) -> None:
    if INSTALL_MARKER.exists():
        return

    subprocess.check_call(
        [
            "python",
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            requirements_url,
        ]
    )
    INSTALL_MARKER.write_text("ok", encoding="utf-8")

    try:
        import IPython

        app = IPython.get_ipython()
        if app is not None:
            app.kernel.do_shutdown(restart=True)
    except Exception:
        raise SystemExit(
            "Dependencies installed. Runtime → Restart session, then Run all again."
        )


def bootstrap_imports() -> None:
    apply_torch_compat_patches()
    clear_cuda_cache()


def load_checkpoint(path: Path = DEFAULT_CHECKPOINT) -> dict[str, Any]:
    if not path.exists():
        return {"completed": {}, "errors": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(payload: dict[str, Any], path: Path = DEFAULT_CHECKPOINT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_scenarios_json(url: str = SCENARIOS_URL) -> dict[str, Any]:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))
