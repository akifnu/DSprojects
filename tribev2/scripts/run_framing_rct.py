#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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
    load_scenarios,
    load_study_metadata,
    run_insilico_framing_experiment,
    save_analysis_report,
)
from tribe_capabilities.inference import load_model, predict_from_text, preload_llama_weights


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Kahneman loss/gain framing RCT inference with TRIBE v2 (text only)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data/framing_rct/scenarios.json")
    parser.add_argument("--n-subjects", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preload-llama", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Limit inference to first N scenario pairs (use in Colab for smoke tests).",
    )
    args = parser.parse_args()

    if not args.protocol.exists():
        print(f"Missing {args.protocol}. Run: python scripts/generate_rct_dataset.py", file=sys.stderr)
        return 1

    config = TribeCapabilitiesConfig.load(args.config)
    scenarios = load_scenarios(args.protocol)
    metadata = load_study_metadata(args.protocol)

    env = check_environment(min_vram_gb=config.min_vram_gb, require_tribev2=True)
    if not env.tribev2_importable:
        print("Install tribev2 first: pip install -r requirements.txt", file=sys.stderr)
        return 1

    if args.preload_llama:
        preload_llama_weights(config)

    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]
        print(f"Running inference on {len(scenarios)} / {metadata.get('n_scenarios', '?')} scenario pairs")

    os.environ["TRIBE_DEVICE"] = args.device
    model = load_model(config, device=args.device)
    predictions = run_insilico_framing_experiment(
        model,
        scenarios,
        peak_timestep=config.visualization_timestep,
        predict_fn=lambda model, text: predict_from_text(model, text),
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

    print(analysis.kahneman_alignment["interpretation"])
    print(f"p = {analysis.kahneman_alignment['primary_p_value_two_sided']:.4f}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
