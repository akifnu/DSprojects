from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tribe_capabilities.benchmarks import _contrast_stats
from tribe_capabilities.config import TribeCapabilitiesConfig
from tribe_capabilities.environment import check_environment
from tribe_capabilities.inference import validate_prediction_shape
from tribe_capabilities.reporting import build_report
from tribe_capabilities.visualization import split_hemispheres


@pytest.fixture
def config() -> TribeCapabilitiesConfig:
    return TribeCapabilitiesConfig.load()


def test_config_loads_expected_vertices(config: TribeCapabilitiesConfig) -> None:
    assert config.model_checkpoint == "facebook/tribev2"
    assert config.expected_vertices == 20484
    assert config.vertices_per_hemisphere == 10242


def test_stimuli_files_exist(config: TribeCapabilitiesConfig) -> None:
    language = config.resolve(config.benchmarks["language_vs_visual"]["stimulus_a"])
    visual = config.resolve(config.benchmarks["language_vs_visual"]["stimulus_b"])
    assert language.exists()
    assert visual.exists()
    assert language.read_text(encoding="utf-8").strip()
    assert visual.read_text(encoding="utf-8").strip()


def test_environment_check_runs() -> None:
    report = check_environment()
    assert report.python_version
    assert isinstance(report.warnings, list)


def test_validate_prediction_shape_accepts_valid_array(config: TribeCapabilitiesConfig) -> None:
    preds = np.random.randn(12, config.expected_vertices).astype(np.float32)
    stats = validate_prediction_shape(preds, expected_vertices=config.expected_vertices)
    assert stats["timesteps"] == 12
    assert stats["vertices"] == config.expected_vertices


def test_validate_prediction_shape_rejects_bad_vertex_count(config: TribeCapabilitiesConfig) -> None:
    preds = np.random.randn(5, 100)
    with pytest.raises(ValueError, match="Expected 20484 vertices"):
        validate_prediction_shape(preds, expected_vertices=config.expected_vertices)


def test_contrast_stats_shape() -> None:
    a = np.ones((10, 20484))
    b = np.zeros((8, 20484))
    stats = _contrast_stats(a, b)
    assert stats["shared_timesteps"] == 8.0
    assert stats["contrast_mean_abs"] == 1.0


def test_split_hemispheres() -> None:
    vector = np.arange(20484, dtype=np.float32)
    left, right = split_hemispheres(vector, vertices_per_hemisphere=10242)
    assert left.shape == (10242,)
    assert right.shape == (10242,)
    assert left[0] == 0
    assert right[0] == 10242


def test_report_builder_serializes_json(config: TribeCapabilitiesConfig) -> None:
    env = check_environment()
    report = build_report(env, [])
    payload = json.loads(json.dumps(report.to_dict()))
    assert "environment" in payload
    assert "summary" in payload


@pytest.mark.gpu
def test_gpu_inference_smoke(config: TribeCapabilitiesConfig) -> None:
    env = check_environment(min_vram_gb=config.min_vram_gb, require_tribev2=True)
    if not env.cuda_available:
        pytest.skip("CUDA GPU not available")

    from tribe_capabilities.inference import load_model, predict_from_text

    model = load_model(config)
    text = config.resolve(config.benchmarks["language_vs_visual"]["stimulus_a"]).read_text(encoding="utf-8")
    preds = predict_from_text(model, text)
    validate_prediction_shape(preds, expected_vertices=config.expected_vertices)
