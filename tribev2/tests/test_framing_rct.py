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
from tribe_capabilities.framing_scenario_bank import build_scenario_bank


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data/framing_rct/scenarios.json"


@pytest.fixture
def scenarios():
    return load_scenarios(PROTOCOL)


def test_scenario_bank_is_large() -> None:
    bank = build_scenario_bank()
    assert len(bank) >= 300


def test_protocol_has_hundreds_of_scenarios(scenarios) -> None:
    assert len(scenarios) >= 300
    domains = {scenario.domain for scenario in scenarios}
    assert len(domains) >= 5


def test_rct_assignments_cover_all_subjects_and_scenarios(scenarios) -> None:
    assignments = generate_rct_assignments(scenarios, n_subjects=200, seed=42)
    assert len(assignments) == 200 * len(scenarios)

    by_subject: dict[str, set[str]] = {}
    for row in assignments:
        by_subject.setdefault(row.subject_id, set()).add(row.scenario_id)

    assert len(by_subject) == 200
    for scenario_ids in by_subject.values():
        assert len(scenario_ids) == len(scenarios)


def test_each_scenario_has_balanced_frames(scenarios) -> None:
    assignments = generate_rct_assignments(scenarios, n_subjects=200, seed=42)
    frame_counts: dict[str, dict[str, int]] = {}

    for row in assignments:
        frame_counts.setdefault(row.scenario_id, {"gain": 0, "loss": 0})
        frame_counts[row.scenario_id][row.frame] += 1

    for scenario in scenarios:
        counts = frame_counts[scenario.scenario_id]
        assert counts["gain"] == 100
        assert counts["loss"] == 100


def test_export_rct_dataset_writes_expected_files(scenarios, tmp_path: Path) -> None:
    assignments = generate_rct_assignments(scenarios[:3], n_subjects=4, seed=7)
    paths = export_rct_dataset(
        scenarios[:3],
        assignments,
        tmp_path,
        seed=7,
        n_subjects=4,
    )

    assert paths["assignments"].exists()
    assert paths["unique_stimuli"].exists()
    assert paths["protocol"].exists()


def test_analyze_framing_predictions_detects_loss_salience() -> None:
    scenarios = load_scenarios(PROTOCOL)[:8]
    predictions: list[ScenarioPrediction] = []
    for scenario in scenarios:
        gain = np.ones((8, 20484), dtype=np.float32)
        loss = np.ones((8, 20484), dtype=np.float32) * 1.5
        for frame, array in ("gain", gain), ("loss", loss):
            metrics = summarize_prediction(array)
            predictions.append(
                ScenarioPrediction(
                    scenario_id=scenario.scenario_id,
                    frame=frame,
                    domain=scenario.domain,
                    prediction=array,
                    metrics=metrics,
                )
            )

    result = analyze_framing_predictions(predictions, study_id="test", n_subjects=200, seed=42)
    assert result.kahneman_alignment["supports_kahneman_direction"] is True
