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


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TRIBE v2 runtime requirements.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    config = TribeCapabilitiesConfig.load(args.config)
    report = check_environment(min_vram_gb=config.min_vram_gb)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("TRIBE v2 environment check")
        print("-" * 32)
        print(f"Python:   {report.python_version}")
        print(f"Platform: {report.platform}")
        print(f"CUDA:     {report.cuda_available}")
        if report.gpu_name:
            print(f"GPU:      {report.gpu_name} ({report.gpu_vram_gb} GB)")
        print(f"NumPy:    {report.numpy_version or 'not installed'}")
        print(f"PyTorch:  {report.torch_version or 'not installed'}")
        print(f"tribev2:  {'installed' if report.tribev2_importable else 'missing'}")
        print(f"HF token: {'present' if report.hf_token_present else 'missing'}")

        if report.warnings:
            print("\nWarnings:")
            for warning in report.warnings:
                print(f"  - {warning}")
        if report.errors:
            print("\nErrors:")
            for error in report.errors:
                print(f"  - {error}")

    return 0 if report.ready_for_inference else 1


if __name__ == "__main__":
    raise SystemExit(main())
