from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

Frame = Literal["gain", "loss"]


@dataclass(frozen=True)
class FramingScenario:
    scenario_id: str
    domain: str
    objective_summary: str
    gain_frame: str
    loss_frame: str
    behavioral_gain_prediction: str
    behavioral_loss_prediction: str

    def text_for(self, frame: Frame) -> str:
        return self.gain_frame if frame == "gain" else self.loss_frame


@dataclass
class TrialAssignment:
    subject_id: str
    trial_id: int
    scenario_id: str
    domain: str
    frame: Frame
    presentation_order: int
    block_id: int
    stimulus_text: str
    stimulus_file: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioPrediction:
    scenario_id: str
    frame: Frame
    domain: str
    prediction: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "frame": self.frame,
            "domain": self.domain,
            "metrics": self.metrics,
            "shape": list(self.prediction.shape),
        }


@dataclass
class FramingAnalysisResult:
    study_id: str
    n_scenarios: int
    n_subjects: int
    random_seed: int
    paired_tests: dict[str, dict[str, float]]
    scenario_comparisons: list[dict[str, Any]]
    kahneman_alignment: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_scenarios(protocol_path: Path) -> list[FramingScenario]:
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    return [FramingScenario(**item) for item in payload["scenarios"]]


def load_study_metadata(protocol_path: Path) -> dict[str, Any]:
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    return {
        key: value
        for key, value in payload.items()
        if key != "scenarios"
    }


def _block_randomize_pairs(
    pairs: list[tuple[int, Frame]],
    rng: np.random.Generator,
) -> list[tuple[int, Frame]]:
    """Swap adjacent trial pairs to balance presentation order."""
    shuffled = pairs.copy()
    for index in range(0, len(shuffled) - 1, 2):
        if rng.random() < 0.5:
            shuffled[index], shuffled[index + 1] = shuffled[index + 1], shuffled[index]
    return shuffled


def generate_rct_assignments(
    scenarios: list[FramingScenario],
    *,
    n_subjects: int = 60,
    seed: int = 42,
) -> list[TrialAssignment]:
    """Generate a within-subjects crossover RCT assignment table.

    Each subject sees every scenario once. Across subjects, each scenario is
    shown in gain and loss frames equally often (50/50 split).
    """
    if n_subjects % 2 != 0:
        raise ValueError("n_subjects must be even for balanced gain/loss crossover.")

    rng = np.random.default_rng(seed)
    assignments: list[TrialAssignment] = []
    trial_counter = 0

    frame_by_subject_scenario: dict[tuple[str, str], Frame] = {}
    for scenario in scenarios:
        subject_ids = [f"SUBJ_{index + 1:03d}" for index in range(n_subjects)]
        gain_subjects = set(rng.choice(subject_ids, size=n_subjects // 2, replace=False))
        for subject_id in subject_ids:
            frame: Frame = "gain" if subject_id in gain_subjects else "loss"
            frame_by_subject_scenario[(subject_id, scenario.scenario_id)] = frame

    for subject_index in range(n_subjects):
        subject_id = f"SUBJ_{subject_index + 1:03d}"
        scenario_order = list(rng.permutation(len(scenarios)))

        trial_pairs: list[tuple[int, Frame]] = [
            (
                scenario_position,
                frame_by_subject_scenario[(subject_id, scenarios[scenario_position].scenario_id)],
            )
            for scenario_position in scenario_order
        ]
        trial_pairs = _block_randomize_pairs(trial_pairs, rng)

        for order, (scenario_position, frame) in enumerate(trial_pairs, start=1):
            scenario = scenarios[scenario_position]
            trial_counter += 1
            block_id = math.ceil(order / 2)
            stimulus_file = f"stimuli/{scenario.scenario_id}_{frame}.txt"
            assignments.append(
                TrialAssignment(
                    subject_id=subject_id,
                    trial_id=trial_counter,
                    scenario_id=scenario.scenario_id,
                    domain=scenario.domain,
                    frame=frame,
                    presentation_order=order,
                    block_id=block_id,
                    stimulus_text=scenario.text_for(frame),
                    stimulus_file=stimulus_file,
                )
            )

    return assignments


def export_rct_dataset(
    scenarios: list[FramingScenario],
    assignments: list[TrialAssignment],
    output_dir: Path,
    *,
    study_metadata: dict[str, Any] | None = None,
    seed: int = 42,
    n_subjects: int = 60,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stimuli_dir = output_dir / "stimuli"
    stimuli_dir.mkdir(exist_ok=True)

    for scenario in scenarios:
        for frame in ("gain", "loss"):
            path = stimuli_dir / f"{scenario.scenario_id}_{frame}.txt"
            path.write_text(scenario.text_for(frame), encoding="utf-8")

    assignments_path = output_dir / "assignments.csv"
    with assignments_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject_id",
                "trial_id",
                "scenario_id",
                "domain",
                "frame",
                "presentation_order",
                "block_id",
                "stimulus_file",
                "stimulus_text",
            ],
        )
        writer.writeheader()
        for row in assignments:
            writer.writerow(row.to_dict())

    unique_trials_path = output_dir / "unique_stimuli.csv"
    seen: set[tuple[str, Frame]] = set()
    with unique_trials_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_id", "domain", "frame", "stimulus_file", "stimulus_text"],
        )
        writer.writeheader()
        for scenario in scenarios:
            for frame in ("gain", "loss"):
                key = (scenario.scenario_id, frame)
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow(
                    {
                        "scenario_id": scenario.scenario_id,
                        "domain": scenario.domain,
                        "frame": frame,
                        "stimulus_file": f"stimuli/{scenario.scenario_id}_{frame}.txt",
                        "stimulus_text": scenario.text_for(frame),
                    }
                )

    protocol = {
        **(study_metadata or {}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "n_subjects": n_subjects,
        "n_scenarios": len(scenarios),
        "n_trials": len(assignments),
        "design_notes": (
            "Within-subjects crossover RCT with block-randomized presentation order. "
            "In-silico inference uses scenario-level paired gain/loss stimuli as "
            "statistical units when testing TRIBE v2."
        ),
        "analysis_unit_for_insilico": "scenario_pair",
        "assignment_file": assignments_path.name,
        "unique_stimuli_file": unique_trials_path.name,
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    return {
        "assignments": assignments_path,
        "unique_stimuli": unique_trials_path,
        "protocol": protocol_path,
        "stimuli_dir": stimuli_dir,
    }


def summarize_prediction(
    prediction: np.ndarray,
    *,
    peak_timestep: int = 5,
) -> dict[str, float]:
    if prediction.ndim != 2:
        raise ValueError(f"Expected 2D prediction, got shape {prediction.shape}")

    timestep = min(peak_timestep, prediction.shape[0] - 1)
    peak_vector = prediction[timestep]
    time_mean = prediction.mean(axis=0)

    return {
        "timesteps": float(prediction.shape[0]),
        "peak_timestep_used": float(timestep),
        "mean_activation": float(np.mean(prediction)),
        "mean_abs_activation": float(np.mean(np.abs(prediction))),
        "peak_mean_activation": float(np.mean(peak_vector)),
        "peak_mean_abs_activation": float(np.mean(np.abs(peak_vector))),
        "activation_l2_norm": float(np.linalg.norm(time_mean)),
        "peak_activation_l2_norm": float(np.linalg.norm(peak_vector)),
        "max_abs_activation": float(np.max(np.abs(prediction))),
    }


def _paired_ttest(sample_a: np.ndarray, sample_b: np.ndarray) -> dict[str, float]:
    if sample_a.shape != sample_b.shape:
        raise ValueError("Paired samples must have the same shape.")

    differences = sample_a - sample_b
    n = len(differences)
    if n < 2:
        return {
            "n_pairs": float(n),
            "mean_difference": float(np.mean(differences)) if n else 0.0,
            "t_statistic": float("nan"),
            "p_value_two_sided": float("nan"),
            "cohens_dz": float("nan"),
        }

    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    t_stat = mean_diff / (std_diff / math.sqrt(n)) if std_diff > 0 else float("inf")
    # Two-sided p-value from Student's t distribution.
    from scipy.stats import t as student_t

    p_value = float(2 * student_t.sf(abs(t_stat), df=n - 1))
    cohens_dz = mean_diff / std_diff if std_diff > 0 else float("inf")

    return {
        "n_pairs": float(n),
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "t_statistic": float(t_stat),
        "p_value_two_sided": p_value,
        "cohens_dz": float(cohens_dz),
    }


def analyze_framing_predictions(
  predictions: list[ScenarioPrediction],
  *,
  study_id: str,
  n_subjects: int,
  seed: int,
  metrics: tuple[str, ...] = (
      "mean_abs_activation",
      "peak_mean_abs_activation",
      "activation_l2_norm",
      "peak_activation_l2_norm",
  ),
) -> FramingAnalysisResult:
    by_scenario: dict[str, dict[Frame, ScenarioPrediction]] = {}
    for item in predictions:
        by_scenario.setdefault(item.scenario_id, {})[item.frame] = item

    scenario_comparisons: list[dict[str, Any]] = []
    metric_arrays: dict[str, dict[str, list[float]]] = {
        metric: {"gain": [], "loss": []} for metric in metrics
    }

    for scenario_id, frames in sorted(by_scenario.items()):
        if "gain" not in frames or "loss" not in frames:
            continue

        gain = frames["gain"]
        loss = frames["loss"]
        min_len = min(gain.prediction.shape[0], loss.prediction.shape[0])
        contrast = loss.prediction[:min_len] - gain.prediction[:min_len]

        comparison = {
            "scenario_id": scenario_id,
            "domain": gain.domain,
            "gain_metrics": gain.metrics,
            "loss_metrics": loss.metrics,
            "loss_minus_gain": {
                metric: loss.metrics[metric] - gain.metrics[metric] for metric in metrics
            },
            "contrast_mean_abs": float(np.mean(np.abs(contrast))),
            "loss_greater_than_gain": {
                metric: loss.metrics[metric] > gain.metrics[metric] for metric in metrics
            },
        }
        scenario_comparisons.append(comparison)

        for metric in metrics:
            metric_arrays[metric]["gain"].append(gain.metrics[metric])
            metric_arrays[metric]["loss"].append(loss.metrics[metric])

    paired_tests = {
        metric: _paired_ttest(
            np.asarray(metric_arrays[metric]["loss"]),
            np.asarray(metric_arrays[metric]["gain"]),
        )
        for metric in metrics
    }

    primary_metric = "mean_abs_activation"
    primary_test = paired_tests[primary_metric]
    n_scenarios = len(scenario_comparisons)
    n_aligned = sum(
        1
        for row in scenario_comparisons
        if row["loss_greater_than_gain"][primary_metric]
    )

    kahneman_alignment = {
        "behavioral_reference": (
            "Kahneman framing: equivalent outcomes are evaluated differently "
            "under gain vs loss wording; losses loom larger than gains."
        ),
        "neural_proxy_tested": (
            "loss_frame cortical magnitude > gain_frame cortical magnitude "
            f"using '{primary_metric}'"
        ),
        "scenarios_with_loss_greater_than_gain": n_aligned,
        "proportion_aligned": n_aligned / n_scenarios if n_scenarios else 0.0,
        "primary_metric": primary_metric,
        "primary_mean_difference_loss_minus_gain": primary_test["mean_difference"],
        "primary_p_value_two_sided": primary_test["p_value_two_sided"],
        "primary_cohens_dz": primary_test["cohens_dz"],
        "supports_kahneman_direction": (
            primary_test["mean_difference"] > 0
            and primary_test["p_value_two_sided"] < 0.05
        ),
        "interpretation": _interpret_alignment(
            primary_test["mean_difference"],
            primary_test["p_value_two_sided"],
            n_aligned,
            n_scenarios,
        ),
    }

    return FramingAnalysisResult(
        study_id=study_id,
        n_scenarios=n_scenarios,
        n_subjects=n_subjects,
        random_seed=seed,
        paired_tests=paired_tests,
        scenario_comparisons=scenario_comparisons,
        kahneman_alignment=kahneman_alignment,
    )


def _interpret_alignment(
    mean_difference: float,
    p_value: float,
    n_aligned: int,
    n_scenarios: int,
) -> str:
    direction = "stronger" if mean_difference > 0 else "weaker"
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    return (
        f"Loss-framed stimuli produced {direction} mean absolute cortical activation "
        f"than gain-framed stimuli across scenario pairs ({significance}, p={p_value:.4f}). "
        f"{n_aligned}/{n_scenarios} scenarios showed higher loss-frame activation, "
        "which is the direction expected if neural magnitude tracks Kahneman-style loss salience."
    )


def run_insilico_framing_experiment(
    model,
    scenarios: list[FramingScenario],
    *,
    peak_timestep: int = 5,
    predict_fn,
) -> list[ScenarioPrediction]:
    results: list[ScenarioPrediction] = []
    for scenario in scenarios:
        for frame in ("gain", "loss"):
            text = scenario.text_for(frame)
            prediction = predict_fn(model, text)
            metrics = summarize_prediction(prediction, peak_timestep=peak_timestep)
            results.append(
                ScenarioPrediction(
                    scenario_id=scenario.scenario_id,
                    frame=frame,
                    domain=scenario.domain,
                    prediction=np.asarray(prediction),
                    metrics=metrics,
                )
            )
    return results


def save_analysis_report(result: FramingAnalysisResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return output_path
