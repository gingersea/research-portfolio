"""Global configuration for physcheck.

Paths and runtime settings shared by all stages (generate / detect / judge /
report). Environment overrides:

- PHYSCHECK_GPU_ID       -> gpu_id
- PHYSCHECK_ARTIFACTS_DIR -> artifacts_dir (absolute or relative to repo root)
- PHYSCHECK_RESULTS_DIR   -> results_dir
- PHYSCHECK_MODELS_DIR    -> models_cache_dir
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    repo_root: Path
    artifacts_dir: Path
    models_cache_dir: Path
    results_dir: Path
    seed_grid: list[int] = field(default_factory=lambda: [0, 1, 2])
    gpu_id: int = 0

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root)
        self.artifacts_dir = Path(self.artifacts_dir)
        self.models_cache_dir = Path(self.models_cache_dir)
        self.results_dir = Path(self.results_dir)
        for d in (self.artifacts_dir, self.models_cache_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, repo_root: str | Path | None = None) -> "Config":
        """Load configuration with environment overrides.

        repo_root defaults to the git repository containing the current
        working directory, discovered via `git rev-parse --show-toplevel`.
        """
        if repo_root is None:
            import subprocess

            repo_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], text=True
            ).strip()
        root = Path(repo_root)
        artifacts = os.environ.get("PHYSCHECK_ARTIFACTS_DIR")
        results = os.environ.get("PHYSCHECK_RESULTS_DIR")
        models = os.environ.get("PHYSCHECK_MODELS_DIR")
        gpu = os.environ.get("PHYSCHECK_GPU_ID")
        cfg = cls(
            repo_root=root,
            artifacts_dir=Path(artifacts) if artifacts else root / "benchmark" / "physcheck" / "artifacts",
            results_dir=Path(results) if results else root / "benchmark" / "results",
            models_cache_dir=Path(models) if models else root / "benchmark" / "physcheck" / "models_cache",
            gpu_id=int(gpu) if gpu else 0,
        )
        if not cfg.artifacts_dir.is_absolute():
            cfg.artifacts_dir = root / cfg.artifacts_dir
        if not cfg.results_dir.is_absolute():
            cfg.results_dir = root / cfg.results_dir
        if not cfg.models_cache_dir.is_absolute():
            cfg.models_cache_dir = root / cfg.models_cache_dir
        cfg.__post_init__()
        return cfg
