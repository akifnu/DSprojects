from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tribe_capabilities.config import TribeCapabilitiesConfig
from tribe_capabilities.inference import predict_from_path, predict_from_text, validate_prediction_shape


@dataclass
class BenchmarkResult:
    name: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contrast_stats(preds_a: np.ndarray, preds_b: np.ndarray) -> dict[str, float]:
    min_len = min(preds_a.shape[0], preds_b.shape[0])
    contrast = preds_a[:min_len] - preds_b[:min_len]
    left = contrast[:, : contrast.shape[1] // 2]
    right = contrast[:, contrast.shape[1] // 2 :]
    return {
        "contrast_mean_abs": float(np.mean(np.abs(contrast))),
        "contrast_max_abs": float(np.max(np.abs(contrast))),
        "left_mean_abs": float(np.mean(np.abs(left))),
        "right_mean_abs": float(np.mean(np.abs(right))),
        "shared_timesteps": float(min_len),
    }


def run_language_vs_visual_benchmark(model, config: TribeCapabilitiesConfig) -> BenchmarkResult:
    benchmark_cfg = config.benchmarks["language_vs_visual"]
    if not benchmark_cfg.get("enabled", True):
        return BenchmarkResult(name="language_vs_visual", status="skipped")

    text_a = config.resolve(benchmark_cfg["stimulus_a"]).read_text(encoding="utf-8")
    text_b = config.resolve(benchmark_cfg["stimulus_b"]).read_text(encoding="utf-8")

    preds_a = predict_from_text(model, text_a)
    preds_b = predict_from_text(model, text_b)
    stats_a = validate_prediction_shape(preds_a, expected_vertices=config.expected_vertices)
    stats_b = validate_prediction_shape(preds_b, expected_vertices=config.expected_vertices)
    contrast = _contrast_stats(preds_a, preds_b)

    return BenchmarkResult(
        name="language_vs_visual",
        status="passed",
        details={
            "condition_a": stats_a,
            "condition_b": stats_b,
            "contrast": contrast,
        },
    )


def run_modality_benchmark(model, config: TribeCapabilitiesConfig, modality: str) -> BenchmarkResult:
    modality_cfg = config.benchmarks["modalities"][modality]
    if not modality_cfg.get("enabled", False):
        return BenchmarkResult(name=f"modality_{modality}", status="skipped")

    stimulus = config.resolve(modality_cfg["stimulus"])
    if not stimulus.exists():
        return BenchmarkResult(
            name=f"modality_{modality}",
            status="skipped",
            details={"reason": f"Missing stimulus file: {stimulus}"},
        )

    if modality == "text":
        preds = predict_from_path(model, text_path=stimulus)
    elif modality == "audio":
        preds = predict_from_path(model, audio_path=stimulus)
    elif modality == "video":
        preds = predict_from_path(model, video_path=stimulus)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    stats = validate_prediction_shape(preds, expected_vertices=config.expected_vertices)
    return BenchmarkResult(
        name=f"modality_{modality}",
        status="passed",
        details={"prediction": stats},
    )


def run_all_benchmarks(model, config: TribeCapabilitiesConfig) -> list[BenchmarkResult]:
    results = [run_language_vs_visual_benchmark(model, config)]
    for modality in ("text", "audio", "video"):
        results.append(run_modality_benchmark(model, config, modality))
    return results
