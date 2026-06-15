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

from tribe_capabilities.config import TribeCapabilitiesConfig
from tribe_capabilities.environment import check_environment
from tribe_capabilities.framing_rct import (
    analyze_framing_predictions,
    export_rct_dataset,
    generate_rct_assignments,
    load_scenarios,
    load_study_metadata,
    run_insilico_framing_experiment,
    save_analysis_report,
)
from tribe_capabilities.inference import load_model, predict_from_text, preload_llama_weights


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and run a Kahneman loss/gain framing RCT with TRIBE v2."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/framing_rct/scenarios.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/framing_rct")
    parser.add_argument("--n-subjects", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-only", action="store_true", help="Only build RCT dataset files.")
    parser.add_argument("--preload-llama", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Limit inference to the first N scenarios (useful for GPU smoke tests).",
    )
    args = parser.parse_args()

    config = TribeCapabilitiesConfig.load(args.config)
    scenarios = load_scenarios(args.protocol)
    metadata = load_study_metadata(args.protocol)
    assignments = generate_rct_assignments(
        scenarios,
        n_subjects=args.n_subjects,
        seed=args.seed,
    )
    paths = export_rct_dataset(
        scenarios,
        assignments,
        args.output_dir,
        study_metadata=metadata,
        seed=args.seed,
        n_subjects=args.n_subjects,
    )

    print("RCT dataset generated")
    print("-" * 32)
    for name, path in paths.items():
        if path.is_file():
            print(f"{name}: {path}")

    if args.generate_only:
        return 0

    env = check_environment(min_vram_gb=config.min_vram_gb, require_tribev2=not args.skip_gpu)
    if args.skip_gpu or not env.cuda_available:
        print("\nSkipping TRIBE inference (no GPU). Dataset is ready for GPU run.")
        return 0

    if not env.tribev2_importable:
        print("tribev2 is not installed. Run scripts/setup.sh first.", file=sys.stderr)
        return 1

    if args.preload_llama:
        preload_llama_weights(config)

    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]
        print(f"Limited to {len(scenarios)} scenario(s) for smoke testing.")

    model = load_model(config)
    predictions = run_insilico_framing_experiment(
        model,
        scenarios,
        peak_timestep=config.visualization_timestep,
        predict_fn=lambda _model, text: predict_from_text(_model, text),
    )

    analysis = analyze_framing_predictions(
        predictions,
        study_id=metadata["study_id"],
        n_subjects=args.n_subjects,
        seed=args.seed,
    )
    report_path = save_analysis_report(
        analysis,
        config.report_output_dir / "framing_rct_analysis.json",
    )

    print("\nFraming RCT analysis")
    print("-" * 32)
    print(analysis.kahneman_alignment["interpretation"])
    print(f"\nPrimary p-value: {analysis.kahneman_alignment['primary_p_value_two_sided']:.4f}")
    print(f"Cohen's dz: {analysis.kahneman_alignment['primary_cohens_dz']:.3f}")
    print(f"Aligned scenarios: {analysis.kahneman_alignment['scenarios_with_loss_greater_than_gain']}/{analysis.n_scenarios}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
