#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tribe_capabilities.framing_rct import export_rct_dataset, generate_rct_assignments, load_scenarios
from tribe_capabilities.framing_scenario_bank import build_scenario_bank


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the full Kahneman framing RCT text dataset.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/framing_rct")
    parser.add_argument("--n-subjects", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-scenarios", type=Path, default=ROOT / "data/framing_rct/scenarios.json")
    args = parser.parse_args()

    bank = build_scenario_bank()
    metadata = {
        "study_id": "kahneman_framing_rct_v1",
        "title": "Loss vs Gain Framing In-Silico RCT (TRIBE v2)",
        "design": "within_subjects_crossover",
        "reference": "Tversky & Kahneman (1981); Kahneman & Tversky (1979) Prospect Theory",
        "primary_hypothesis": (
            "Loss-framed text stimuli produce systematically different cortical activation "
            "patterns than objectively equivalent gain-framed text stimuli."
        ),
        "behavioral_benchmark": (
            "Humans show framing effects: risk-averse in gain domain, risk-seeking in loss domain."
        ),
        "kahneman_directional_prediction": "loss_frame_mean_abs_activation > gain_frame_mean_abs_activation",
        "scenarios": bank,
    }

    args.write_scenarios.parent.mkdir(parents=True, exist_ok=True)
    args.write_scenarios.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    scenarios = load_scenarios(args.write_scenarios)
    assignments = generate_rct_assignments(scenarios, n_subjects=args.n_subjects, seed=args.seed)
    paths = export_rct_dataset(
        scenarios,
        assignments,
        args.output_dir,
        study_metadata={k: v for k, v in metadata.items() if k != "scenarios"},
        seed=args.seed,
        n_subjects=args.n_subjects,
    )

    print("RCT dataset built")
    print(f"  scenarios: {len(scenarios)}")
    print(f"  unique texts: {len(scenarios) * 2}")
    print(f"  subjects: {args.n_subjects}")
    print(f"  trials: {len(assignments)}")
    for name, path in paths.items():
        if path.is_file():
            print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
