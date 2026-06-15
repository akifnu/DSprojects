from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tribe_capabilities.config import TribeCapabilitiesConfig


def configure_huggingface(config: TribeCapabilitiesConfig) -> None:
    timeout = str(config.download_timeout_seconds)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", timeout)
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", timeout)


def preload_llama_weights(config: TribeCapabilitiesConfig) -> Path:
    from huggingface_hub import snapshot_download

    cache_dir = config.cache_folder / "llama"
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.llama_repo,
        cache_dir=str(cache_dir),
        ignore_patterns=["*.bin"],
    )
    return cache_dir


def load_model(config: TribeCapabilitiesConfig):
    from tribev2 import TribeModel

    configure_huggingface(config)
    config.cache_folder.mkdir(parents=True, exist_ok=True)
    return TribeModel.from_pretrained(
        config.model_checkpoint,
        cache_folder=str(config.cache_folder),
    )


def text_file_to_events(model, text_path: Path):
    return model.get_events_dataframe(text_path=str(text_path))


def write_text_tempfile(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    try:
        handle.write(text.strip())
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        return Path(handle.name)
    except Exception:
        handle.close()
        if os.path.exists(handle.name):
            os.unlink(handle.name)
        raise


def predict_from_text(model, text: str) -> np.ndarray:
    path = write_text_tempfile(text)
    try:
        events = model.get_events_dataframe(text_path=str(path))
        preds, _segments = model.predict(events=events, verbose=False)
        return np.asarray(preds)
    finally:
        if path.exists():
            path.unlink()


def predict_from_path(model, *, text_path: Path | None = None, audio_path: Path | None = None, video_path: Path | None = None) -> np.ndarray:
    kwargs: dict[str, str] = {}
    if text_path is not None:
        kwargs["text_path"] = str(text_path)
    if audio_path is not None:
        kwargs["audio_path"] = str(audio_path)
    if video_path is not None:
        kwargs["video_path"] = str(video_path)

    events = model.get_events_dataframe(**kwargs)
    preds, _segments = model.predict(events=events, verbose=False)
    return np.asarray(preds)


def validate_prediction_shape(
    preds: np.ndarray,
    *,
    expected_vertices: int,
) -> dict[str, Any]:
    if preds.ndim != 2:
        raise ValueError(f"Expected 2D predictions, got shape {preds.shape}")

    timesteps, vertices = preds.shape
    if vertices != expected_vertices:
        raise ValueError(
            f"Expected {expected_vertices} vertices, got {vertices} in shape {preds.shape}"
        )

    finite_ratio = float(np.isfinite(preds).mean())
    if finite_ratio < 1.0:
        raise ValueError(f"Predictions contain non-finite values ({finite_ratio:.2%} finite)")

    return {
        "shape": list(preds.shape),
        "timesteps": timesteps,
        "vertices": vertices,
        "mean_activation": float(np.mean(preds)),
        "std_activation": float(np.std(preds)),
        "max_abs_activation": float(np.max(np.abs(preds))),
        "finite_ratio": finite_ratio,
    }
