from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TribeCapabilitiesConfig:
    root: Path
    model_checkpoint: str
    cache_folder: Path
    min_vram_gb: float
    recommended_vram_gb: float
    llama_repo: str
    download_timeout_seconds: int
    visualization_timestep: int
    expected_vertices: int
    vertices_per_hemisphere: int
    report_output_dir: Path
    benchmarks: dict[str, Any]

    @classmethod
    def load(cls, config_path: Path | None = None) -> "TribeCapabilitiesConfig":
        root = Path(__file__).resolve().parents[2]
        path = config_path or root / "config" / "default.yaml"
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        return cls(
            root=root,
            model_checkpoint=raw["model"]["checkpoint"],
            cache_folder=root / raw["model"]["cache_folder"],
            min_vram_gb=float(raw["hardware"]["min_vram_gb"]),
            recommended_vram_gb=float(raw["hardware"]["recommended_vram_gb"]),
            llama_repo=raw["huggingface"]["llama_repo"],
            download_timeout_seconds=int(raw["huggingface"]["download_timeout_seconds"]),
            visualization_timestep=int(raw["inference"]["visualization_timestep"]),
            expected_vertices=int(raw["inference"]["expected_vertices"]),
            vertices_per_hemisphere=int(raw["inference"]["vertices_per_hemisphere"]),
            report_output_dir=root / raw["reporting"]["output_dir"],
            benchmarks=raw["benchmarks"],
        )

    def resolve(self, relative_path: str) -> Path:
        return (self.root / relative_path).resolve()
