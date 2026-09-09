"""Typed, environment-backed settings for reproducible experiment runs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


@dataclass(frozen=True)
class ExperimentPaths:
    """Filesystem locations; source data remains external to this repository."""

    root: Path
    latent_hatred_dir: Path
    mami_csv: Path
    conceptnet_path: Path
    latent_context_csv: Path
    mami_context_csv: Path
    artifact_dir: Path
    latent_base_csv: Path | None = None

    @classmethod
    def from_environment(cls, root: Path | None = None) -> "ExperimentPaths":
        resolved_root = _path_from_env("HSD_REPO_ROOT", root or Path.cwd()).resolve()
        return cls(
            root=resolved_root,
            latent_hatred_dir=_path_from_env("HSD_LATENT_HATRED_DIR", resolved_root / "implicit-hate-corpus"),
            mami_csv=_path_from_env("HSD_MAMI_CSV", resolved_root / "data" / "mami" / "MAMI_Base.csv"),
            conceptnet_path=_path_from_env(
                "HSD_CONCEPTNET_PATH", resolved_root / "data" / "numberbatch-en-19.08.txt" / "numberbatch-en.txt"
            ),
            latent_context_csv=_path_from_env(
                "HSD_LATENT_CONTEXT_CSV", resolved_root / "data" / "latent hatred" / "LH_Context.csv"
            ),
            mami_context_csv=_path_from_env("HSD_MAMI_CONTEXT_CSV", resolved_root / "data" / "mami" / "MAMI_Context.csv"),
            artifact_dir=_path_from_env("HSD_ARTIFACT_DIR", resolved_root / "artifacts"),
            latent_base_csv=_path_from_env("HSD_LATENT_BASE_CSV", resolved_root / "data" / "latent hatred" / "LH_Base.csv"),
        )

    def create_artifact_directories(self) -> None:
        """Create only explicit local run-artifact directories."""
        (self.artifact_dir / "manifests").mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "requests").mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "results").mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.2
    split_seed: int = 42
    runs: int = 5
    epochs: int = 200
    learning_rate: float = 0.001
    hidden_dimension: int = 512
    device: str = field(default_factory=lambda: os.environ.get("HSD_DEVICE", "cuda"))

    def seed_for_run(self, run_index: int) -> int:
        if run_index < 0:
            raise ValueError("run_index must be non-negative")
        return self.split_seed + run_index


@dataclass(frozen=True)
class ModelConfig:
    sentence_model: str = "sentence-transformers/all-mpnet-base-v2"
    message_max_length: int = 256


@dataclass(frozen=True)
class VertexConfig:
    project_id: str = field(default_factory=lambda: os.environ.get("HSD_GCP_PROJECT", "hsd-general"))
    bucket_name: str = field(default_factory=lambda: os.environ.get("HSD_GCS_BUCKET", "hsd-bucket"))
    location: str = field(default_factory=lambda: os.environ.get("HSD_VERTEX_LOCATION", "europe-west4"))
    model_id: str = field(default_factory=lambda: os.environ.get("HSD_GEMINI_MODEL", "gemini-2.5-flash-lite"))
    input_prefix: str = field(default_factory=lambda: os.environ.get("HSD_GCS_INPUT_PREFIX", "contextual-hsd/requests"))
    output_prefix: str = field(default_factory=lambda: os.environ.get("HSD_GCS_OUTPUT_PREFIX", "contextual-hsd/predictions"))

    @property
    def bucket_uri(self) -> str:
        return f"gs://{self.bucket_name}"


@dataclass(frozen=True)
class ExperimentConfig:
    paths: ExperimentPaths
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    vertex: VertexConfig = field(default_factory=VertexConfig)

    @classmethod
    def from_environment(cls, root: Path | None = None) -> "ExperimentConfig":
        return cls(paths=ExperimentPaths.from_environment(root))
