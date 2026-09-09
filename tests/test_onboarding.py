from pathlib import Path

import pytest

from src.config import ExperimentPaths
from src.onboarding import validate_run_inputs


def _paths(tmp_path: Path) -> ExperimentPaths:
    return ExperimentPaths(
        root=tmp_path,
        latent_hatred_dir=tmp_path / "latent",
        mami_csv=tmp_path / "mami.csv",
        conceptnet_path=tmp_path / "numberbatch.txt",
        latent_context_csv=tmp_path / "latent-context.csv",
        mami_context_csv=tmp_path / "mami-context.csv",
        artifact_dir=tmp_path / "artifacts",
    )


def test_preflight_lists_missing_latent_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Latent Hatred stage 1 IDs"):
        validate_run_inputs(_paths(tmp_path), "latent", "zero-context")


def test_preflight_accepts_required_latent_files_and_context(tmp_path):
    paths = _paths(tmp_path)
    paths.latent_hatred_dir.mkdir()
    for stage in (1, 2, 3):
        (paths.latent_hatred_dir / f"implicit_hate_v1_stg{stage}.tsv").touch()
        (paths.latent_hatred_dir / f"implicit_hate_v1_stg{stage}_posts.tsv").touch()
    paths.latent_context_csv.touch()

    validate_run_inputs(paths, "latent", "embed-concat")


def test_preflight_rejects_unknown_choices(tmp_path):
    with pytest.raises(ValueError, match="DATASET"):
        validate_run_inputs(_paths(tmp_path), "other", "zero-context")
    with pytest.raises(ValueError, match="EXPERIMENT"):
        validate_run_inputs(_paths(tmp_path), "latent", "other")
