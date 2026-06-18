from __future__ import annotations

import gc
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from tribe_capabilities.config import TribeCapabilitiesConfig


def configure_huggingface(config: TribeCapabilitiesConfig) -> None:
    timeout = str(config.download_timeout_seconds)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", timeout)
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", timeout)


def apply_torch_compat_patches() -> None:
    """Patch missing torch dtypes referenced by some transformers builds."""
    import torch

    if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e4m3fn"):
        torch.float8_e8m0fnu = torch.float8_e4m3fn  # type: ignore[attr-defined]


def clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


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


def load_model(config: TribeCapabilitiesConfig, *, device: str | None = None):
    apply_torch_compat_patches()
    from tribev2 import TribeModel

    configure_huggingface(config)
    config.cache_folder.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = os.environ.get("TRIBE_DEVICE", "auto")
    return TribeModel.from_pretrained(
        config.model_checkpoint,
        cache_folder=str(config.cache_folder),
        device=device,
    )


def build_text_events(
    text: str,
    *,
    word_duration: float = 0.35,
    timeline: str = "default",
    subject: str = "default",
) -> pd.DataFrame:
    """Build word-level events from plain text without TTS or audio."""
    from neuralset.events.transforms import (
        AddContextToWords,
        AddSentenceToWords,
        AddText,
        RemoveMissing,
    )
    from neuralset.events.utils import standardize_events

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Text is empty")

    words = re.findall(r"\S+", cleaned)
    rows: list[dict[str, Any]] = []
    cursor = 0.0
    for word in words:
        rows.append(
            {
                "type": "Word",
                "text": word,
                "start": cursor,
                "duration": word_duration,
                "timeline": timeline,
                "subject": subject,
            }
        )
        cursor += word_duration

    rows.append(
        {
            "type": "Text",
            "text": cleaned,
            "start": 0.0,
            "duration": cursor,
            "timeline": timeline,
            "subject": subject,
        }
    )

    events = standardize_events(pd.DataFrame(rows))
    for transform in (
        AddText(),
        AddSentenceToWords(max_unmatched_ratio=0.05),
        AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
        RemoveMissing(),
    ):
        events = transform(events)
    return standardize_events(events)


def predict_from_events(model, events: pd.DataFrame) -> np.ndarray:
    preds, _segments = model.predict(events=events, verbose=False)
    return np.asarray(preds)


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
    events = build_text_events(text)
    return predict_from_events(model, events)


def predict_from_path(
    model,
    *,
    text_path: Path | None = None,
    audio_path: Path | None = None,
    video_path: Path | None = None,
) -> np.ndarray:
    kwargs: dict[str, str] = {}
    if text_path is not None:
        text = text_path.read_text(encoding="utf-8")
        return predict_from_text(model, text)
    if audio_path is not None:
        kwargs["audio_path"] = str(audio_path)
    if video_path is not None:
        kwargs["video_path"] = str(video_path)

    events = model.get_events_dataframe(**kwargs)
    return predict_from_events(model, events)


def predict_from_text_resilient(
    model,
    text: str,
    *,
    max_attempts: int = 3,
    retry_delay_seconds: float = 2.0,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return predict_from_text(model, text)
        except Exception as exc:  # noqa: BLE001 - Colab needs broad retries
            last_error = exc
            clear_cuda_cache()
            if on_retry is not None:
                on_retry(attempt, exc)
            if attempt >= max_attempts:
                break
            time.sleep(retry_delay_seconds * attempt)
    assert last_error is not None
    raise last_error


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
