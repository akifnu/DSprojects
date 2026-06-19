from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tribe_capabilities.config import TribeCapabilitiesConfig
from tribe_capabilities.environment import check_environment
from tribe_capabilities.inference import (
    apply_torch_compat_patches,
    clear_cuda_cache,
    load_model_colab,
    predict_batch_texts,
    predict_from_text_resilient,
)

INSTALL_MARKER = Path("/content/.tribev2_colab_ready_v6")
DEFAULT_CHECKPOINT = Path("/content/framing_rct_checkpoint.json")
DEFAULT_CACHE = Path("/content/tribe_cache")
REPO_DIR = Path("/content/DSprojects")
REPO_URL = "https://github.com/akifnu/DSprojects.git"

# Three canonical Kahneman pairs — 6 texts total, finishes much faster than 12+.
QUICK_PAIRS: list[dict[str, str]] = [
    {
        "scenario_id": "asian_disease",
        "domain": "health",
        "gain_frame": (
            "Program A will save 200 people for certain. "
            "Program B has a one-third probability that all 600 people will be saved."
        ),
        "loss_frame": (
            "Program C will result in 400 people dying for certain. "
            "Program D has a one-third probability that nobody will die."
        ),
    },
    {
        "scenario_id": "surgery",
        "domain": "health",
        "gain_frame": (
            "The operation has a 90 percent success rate. "
            "Nine out of ten patients recover fully."
        ),
        "loss_frame": (
            "The operation has a 10 percent failure rate. "
            "One out of ten patients do not survive."
        ),
    },
    {
        "scenario_id": "credit_card",
        "domain": "financial",
        "gain_frame": "Paying cash gives you a 1 dollar discount compared to the credit card price.",
        "loss_frame": "Paying by credit card adds a 1 dollar surcharge compared to the cash price.",
    },
]


@dataclass
class GpuReport:
    name: str | None
    vram_gb: float | None
    cuda_available: bool
    warnings: list[str]


@dataclass
class QuickDemoResult:
    rows: list[dict[str, Any]]
    errors: list[dict[str, str]]
    checkpoint_path: Path


def query_gpu() -> GpuReport:
    report = check_environment(min_vram_gb=0.0, require_tribev2=False)
    warnings = list(report.warnings)
    if report.gpu_vram_gb is not None and report.gpu_vram_gb < 20:
        warnings.append(
            f"Detected {report.gpu_vram_gb:.1f} GB VRAM. "
            "Use Colab Pro A100 if possible; T4 may OOM even in quick mode."
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


def install_colab_dependencies() -> None:
    if INSTALL_MARKER.exists():
        return

    if not REPO_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "tribev2/requirements-colab.txt")])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR / "tribev2")])
    INSTALL_MARKER.write_text("ok", encoding="utf-8")

    import IPython

    app = IPython.get_ipython()
    if app is not None:
        app.kernel.do_shutdown(restart=True)
    raise SystemExit("Dependencies installed — runtime is restarting. Click Run all again.")


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


def summarize_prediction(preds: np.ndarray, peak_timestep: int = 5) -> dict[str, float]:
    t = min(peak_timestep, preds.shape[0] - 1)
    return {
        "timesteps": float(preds.shape[0]),
        "mean_abs": float(np.mean(np.abs(preds))),
        "peak_abs": float(np.mean(np.abs(preds[t]))),
    }


def run_quick_demo(
    *,
    cache_dir: Path = DEFAULT_CACHE,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    peak_timestep: int = 5,
    use_batch: bool = True,
) -> QuickDemoResult:
    """End-to-end 3-pair demo: login, load text-only model, predict, return stats."""
    bootstrap_imports()
    login_huggingface()

    gpu = query_gpu()
    print(f"GPU: {gpu.name} | VRAM: {gpu.vram_gb} GB | CUDA: {gpu.cuda_available}")
    for warning in gpu.warnings:
        print("⚠", warning)
    if not gpu.cuda_available:
        raise RuntimeError("Enable a GPU runtime before continuing.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = REPO_DIR / "tribev2/config/colab.yaml"
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[2] / "config/colab.yaml"
    config = TribeCapabilitiesConfig.load(config_path)
    from dataclasses import replace

    config = replace(config, cache_folder=cache_dir)

    ckpt = load_checkpoint(checkpoint_path)
    pending: list[tuple[str, str, str, str]] = []
    for scenario in QUICK_PAIRS:
        for frame in ("gain", "loss"):
            key = f"{scenario['scenario_id']}_{frame}"
            if key not in ckpt["completed"]:
                pending.append((key, scenario[f"{frame}_frame"], scenario["scenario_id"], frame))

    print("Loading TRIBE v2 (text encoder only — skips audio/video models)...")
    model = load_model_colab(config, device="cuda")
    print("Model ready.")

    if pending and use_batch:
        try:
            batch_items = [(key, text) for key, text, _sid, _frame in pending]
            print(f"Running batched inference for {len(batch_items)} texts in one pass...")
            outputs = predict_batch_texts(model, batch_items)
            for key, text, _sid, frame in pending:
                if key not in outputs:
                    ckpt["errors"].append({"key": key, "error": "missing batch output"})
                    continue
                stats = summarize_prediction(outputs[key], peak_timestep)
                ckpt["completed"][key] = stats
                save_checkpoint(ckpt, checkpoint_path)
                print(f"[ok] {key} peak={stats['peak_abs']:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"Batch failed ({exc}); falling back to one-by-one...")
            use_batch = False

    if pending and not use_batch:
        for key, text, _sid, frame in pending:
            if key in ckpt["completed"]:
                continue
            try:
                preds = predict_from_text_resilient(
                    model,
                    text,
                    timeline=key,
                    max_attempts=2,
                    on_retry=lambda attempt, err: print(
                        f"  retry {attempt} for {key}: {type(err).__name__}"
                    ),
                )
                stats = summarize_prediction(preds, peak_timestep)
                ckpt["completed"][key] = stats
                save_checkpoint(ckpt, checkpoint_path)
                print(f"[ok] {key} peak={stats['peak_abs']:.4f}")
            except Exception as exc:  # noqa: BLE001
                ckpt["errors"].append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
                save_checkpoint(ckpt, checkpoint_path)
                print(f"[FAIL] {key}: {exc}")
            finally:
                clear_cuda_cache()

    rows: list[dict[str, Any]] = []
    for scenario in QUICK_PAIRS:
        row: dict[str, Any] = {
            "id": scenario["scenario_id"],
            "domain": scenario["domain"],
        }
        complete = True
        for frame in ("gain", "loss"):
            key = f"{scenario['scenario_id']}_{frame}"
            stats = ckpt["completed"].get(key)
            if stats is None:
                complete = False
                break
            row[f"{frame}_mean_abs"] = stats["mean_abs"]
            row[f"{frame}_peak_abs"] = stats["peak_abs"]
        if complete:
            row["loss_minus_gain"] = row["loss_mean_abs"] - row["gain_mean_abs"]
            rows.append(row)
            print(f"{row['id']:16s}  loss-gain = {row['loss_minus_gain']:+.4f}")

    return QuickDemoResult(rows=rows, errors=ckpt["errors"], checkpoint_path=checkpoint_path)
