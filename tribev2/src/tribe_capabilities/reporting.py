from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tribe_capabilities.benchmarks import BenchmarkResult
from tribe_capabilities.environment import EnvironmentReport


@dataclass
class CapabilityReport:
    created_at: str
    environment: dict[str, Any]
    benchmarks: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = output_dir / f"capability_report_{timestamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def build_report(
    environment: EnvironmentReport,
    benchmark_results: list[BenchmarkResult],
) -> CapabilityReport:
    summary = {"passed": 0, "failed": 0, "skipped": 0}
    for result in benchmark_results:
        summary[result.status] = summary.get(result.status, 0) + 1

    return CapabilityReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        environment=environment.to_dict(),
        benchmarks=[result.to_dict() for result in benchmark_results],
        summary=summary,
    )
