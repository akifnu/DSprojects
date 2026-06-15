from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tribe_capabilities.framing_rct import (
    ScenarioPrediction,
    analyze_framing_predictions,
    export_rct_dataset,
    generate_rct_assignments,
    load_scenarios,
    summarize_prediction,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/framing_rct/scenarios.json"


@pytest.fixture
def scenarios():
    return load_scenarios(PROTOCOL)


def test_protocol_has_eight_balanced_scenarios(scenarios) -> None:
    assert len(scenarios) == 8
    domains = {scenario.domain for scenario in scenarios}
    assert len(domains) >= 4


def test_rct_assignments_cover_all_subjects_and_scenarios(scenarios) -> None:
    assignments = generate_rct_assignments(scenarios, n_subjects=60, seed=42)
    assert len(assignments) == 60 * len(scenarios)

    by_subject: dict[str, set[str]] = {}
    for row in assignments:
        by_subject.setdefault(row.subject_id, set()).add(row.scenario_id)

    assert len(by_subject) == 60
    for scenario_ids in by_subject.values():
        assert len(scenario_ids) == len(scenarios)


def test_each_scenario_has_both_frames_across_subjects(scenarios) -> None:
    assignments = generate_rct_assignments(scenarios, n_subjects=60, seed=42)
    frame_counts: dict[str, dict[str, int]] = {}

    for row in assignments:
        frame_counts.setdefault(row.scenario_id, {"gain": 0, "loss": 0})
        frame_counts[row.scenario_id][row.frame] += 1

    for scenario in scenarios:
        counts = frame_counts[scenario.scenario_id]
        assert counts["gain"] == 30
        assert counts["loss"] == 30


def test_export_rct_dataset_writes_expected_files(scenarios, tmp_path: Path) -> None:
    assignments = generate_rct_assignments(scenarios, n_subjects=4, seed=7)
    paths = export_rct_dataset(
        scenarios,
        assignments,
        tmp_path,
        seed=7,
        n_subjects=4,
    )

    assert paths["assignments"].exists()
    assert paths["unique_stimuli"].exists()
    assert paths["protocol"].exists()
    assert (paths["stimuli_dir"] / "asian_disease_gain.txt").exists()
    assert (paths["stimuli_dir"] / "asian_disease_loss.txt").exists()

    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    assert protocol["n_subjects"] == 4
    assert protocol["random_seed"] == 7


def test_summarize_prediction_metrics() -> None:
    prediction = np.random.randn(10, 20484).astype(np.float32)
    metrics = summarize_prediction(prediction, peak_timestep=5)
    assert metrics["timesteps"] == 10.0
    assert metrics["peak_timestep_used"] == 5.0
    assert metrics["mean_abs_activation"] >= 0


def test_analyze_framing_predictions_detects_loss_salience(scenarios) -> None:
    predictions: list[ScenarioPrediction] = []
    for scenario in scenarios:
        gain = np.ones((8, 20484), dtype=np.float32)
        loss = np.ones((8, 20484), dtype=np.float32) * 1.5
        gain_metrics = summarize_prediction(gain)
        loss_metrics = summarize_prediction(loss)
        predictions.append(
            ScenarioPrediction(
                scenario_id=scenario.scenario_id,
                frame="gain",
                domain=scenario.domain,
                prediction=gain,
                metrics=gain_metrics,
            )
        )
        predictions.append(
            ScenarioPrediction(
                scenario_id=scenario.scenario_id,
                frame="loss",
                domain=scenario.domain,
                prediction=loss,
                metrics=loss_metrics,
            )
        )

    result = analyze_framing_predictions(
        predictions,
        study_id="test",
        n_subjects=60,
        seed=42,
    )

    assert result.n_scenarios == len(scenarios)
    assert result.kahneman_alignment["supports_kahneman_direction"] is True
    assert result.paired_tests["mean_abs_activation"]["mean_difference"] > 0
    assert result.paired_tests["mean_abs_activation"]["p_value_two_sided"] < 0.05


def test_analyze_framing_predictions_no_effect(scenarios) -> None:
    predictions: list[ScenarioPrediction] = []
    for scenario in scenarios:
        vector = np.random.randn(8, 20484).astype(np.float32)
        metrics = summarize_prediction(vector)
        for frame in ("gain", "loss"):
            predictions.append(
                ScenarioPrediction(
                    scenario_id=scenario.scenario_id,
                    frame=frame,
                    domain=scenario.domain,
                    prediction=vector,
                    metrics=metrics,
                )
            )

    result = analyze_framing_predictions(
        predictions,
        study_id="test",
        n_subjects=60,
        seed=42,
    )

    assert result.kahneman_alignment["supports_kahneman_direction"] is False
