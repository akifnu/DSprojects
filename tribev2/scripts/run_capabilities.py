#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tribe_capabilities.benchmarks import run_all_benchmarks
from tribe_capabilities.config import TribeCapabilitiesConfig
from tribe_capabilities.environment import check_environment
from tribe_capabilities.inference import load_model, preload_llama_weights
from tribe_capabilities.reporting import build_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TRIBE v2 capability benchmarks against facebook/tribev2."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preload-llama", action="store_true", help="Cache LLaMA 3.2 weights before inference.")
    parser.add_argument("--skip-gpu", action="store_true", help="Only run environment checks.")
    parser.add_argument("--visualize", action="store_true", help="Save an HTML brain snapshot for the first benchmark.")
    args = parser.parse_args()

    config = TribeCapabilitiesConfig.load(args.config)
    env = check_environment(min_vram_gb=config.min_vram_gb, require_tribev2=not args.skip_gpu)

    print("Environment")
    print("-" * 32)
    print(f"CUDA available: {env.cuda_available}")
    if env.gpu_name:
        print(f"GPU: {env.gpu_name} ({env.gpu_vram_gb} GB)")
    for warning in env.warnings:
        print(f"warning: {warning}")

    if args.skip_gpu or not env.cuda_available:
        report = build_report(env, [])
        path = report.save(config.report_output_dir)
        print(f"\nSkipped GPU benchmarks. Wrote report to {path}")
        return 0

    if not env.tribev2_importable:
        print("tribev2 is not installed. Run scripts/setup.sh first.", file=sys.stderr)
        return 1

    if args.preload_llama:
        print("Pre-downloading LLaMA 3.2-3B weights...")
        cache_dir = preload_llama_weights(config)
        print(f"Cached at {cache_dir}")

    print("Loading facebook/tribev2...")
    model = load_model(config)

    print("Running capability benchmarks...")
    results = run_all_benchmarks(model, config)
    report = build_report(env, results)
    report_path = report.save(config.report_output_dir)

    print("\nBenchmark summary")
    print("-" * 32)
    for result in results:
        line = f"{result.name}: {result.status}"
        if result.error:
            line += f" ({result.error})"
        print(line)
    print(f"\nReport saved to {report_path}")

    if args.visualize:
        from tribe_capabilities.visualization import save_brain_snapshot_html

        language_result = next((r for r in results if r.name == "language_vs_visual"), None)
        if language_result and language_result.status == "passed":
            # Re-run a short text prediction for visualization.
            from tribe_capabilities.inference import predict_from_text

            text = config.resolve(config.benchmarks["language_vs_visual"]["stimulus_a"]).read_text(encoding="utf-8")
            preds = predict_from_text(model, text)
            timestep = min(config.visualization_timestep, preds.shape[0] - 1)
            output = config.report_output_dir / "language_brain_snapshot.html"
            save_brain_snapshot_html(
                preds[timestep],
                output,
                vertices_per_hemisphere=config.vertices_per_hemisphere,
                title="Language stimulus",
            )
            print(f"Visualization saved to {output}")

    failed = [result for result in results if result.status == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
