"""Small preflight checks used by the guided orchestrator notebook."""

from __future__ import annotations

from pathlib import Path

from .config import ExperimentPaths
from .experiments import EXPERIMENTS


def validate_run_inputs(paths: ExperimentPaths, dataset: str, experiment: str) -> None:
    """Raise one clear error listing the files needed for a selected run."""
    if dataset not in {"latent", "mami"}:
        raise ValueError("DATASET must be 'latent' or 'mami'")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"EXPERIMENT must be one of: {', '.join(EXPERIMENTS)}")

    required: list[tuple[str, Path]] = []
    if dataset == "latent" and paths.latent_base_csv and paths.latent_base_csv.exists():
        required.append(("Latent Hatred base CSV", paths.latent_base_csv))
    elif dataset == "latent":
        required += [
            (f"Latent Hatred stage {stage} IDs", paths.latent_hatred_dir / f"implicit_hate_v1_stg{stage}.tsv")
            for stage in (1, 2, 3)
        ]
        required += [
            (f"Latent Hatred stage {stage} posts", paths.latent_hatred_dir / f"implicit_hate_v1_stg{stage}_posts.tsv")
            for stage in (1, 2, 3)
        ]
    else:
        required.append(("MAMI CSV", paths.mami_csv))

    if experiment in {"append-embed", "embed-concat", "context-embed"}:
        required.append(("generated context CSV", paths.latent_context_csv if dataset == "latent" else paths.mami_context_csv))
    if experiment == "conceptnet":
        required.append(("ConceptNet Numberbatch vectors", paths.conceptnet_path))

    missing = [(name, path) for name, path in required if not path.exists()]
    if missing:
        details = "\n".join(f"- {name}: {path}" for name, path in missing)
        raise FileNotFoundError(
            "This run is not ready yet. Attach the required Kaggle dataset(s), then set the matching path in the "
            f"notebook configuration cell. Missing files:\n{details}"
        )
