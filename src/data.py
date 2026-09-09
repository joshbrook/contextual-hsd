"""Dataset loading, validation, and deterministic split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import ExperimentPaths, TrainingConfig

MAMI_LABEL_COLUMNS = ("shaming", "stereotype", "objectification", "violence")


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path | str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(sorted(missing))}")


def _read_latent_stage(directory: Path, stage: int) -> pd.DataFrame:
    ids = pd.read_csv(directory / f"implicit_hate_v1_stg{stage}.tsv", sep="\t")
    posts = pd.read_csv(directory / f"implicit_hate_v1_stg{stage}_posts.tsv", sep="\t")
    _require_columns(ids, ["ID"], directory)
    _require_columns(posts, ["post"], directory)
    if len(ids) != len(posts):
        raise ValueError(f"Latent Hatred stage {stage} ID and post files have different row counts")
    posts = posts.copy()
    posts["ID"] = ids["ID"].astype(str).to_numpy()
    return posts


def load_latent_hatred(paths: ExperimentPaths) -> pd.DataFrame:
    """Load the included base CSV, or recreate it from the three raw stages."""
    if paths.latent_base_csv and paths.latent_base_csv.exists():
        combined = pd.read_csv(paths.latent_base_csv)
    else:
        stage_frames = [_read_latent_stage(paths.latent_hatred_dir, stage) for stage in (1, 2, 3)]
        combined = stage_frames[0]
        for index, frame in enumerate(stage_frames[1:], start=2):
            combined = combined.merge(frame, on="ID", how="outer", suffixes=("", f"_{index}"))

        post_columns = [column for column in combined.columns if column == "post" or column.startswith("post_")]
        if not post_columns:
            raise ValueError("Latent Hatred source files did not produce a post column")
        # The original notebook treats the stage-one text as canonical.
        combined["post"] = combined[post_columns].bfill(axis=1).iloc[:, 0]
    required = ["ID", "post", "class", "target", "implied_statement", "implicit_class", "extra_implicit_class"]
    _require_columns(combined, required, paths.latent_base_csv or paths.latent_hatred_dir)
    frame = combined[required].dropna(thresh=2).copy()
    frame["binary_class"] = frame["class"].where(frame["class"].eq("not_hate"), "hate")
    return frame.drop(columns=["target", "implied_statement", "extra_implicit_class"])


def load_mami(paths: ExperimentPaths) -> pd.DataFrame:
    """Load MAMI and preserve the textualised-meme representation."""
    frame = pd.read_csv(paths.mami_csv).copy()
    required = ["meme id", "extracted text", "image description", "misogyny", *MAMI_LABEL_COLUMNS]
    _require_columns(frame, required, paths.mami_csv)
    frame["post"] = (
        frame["extracted text"].fillna("").astype(str)
        + " "
        + frame["image description"].fillna("").astype(str)
    ).str.replace("\n", " ", regex=False)
    return frame


def create_splits(
    frame: pd.DataFrame,
    stratify_column: str,
    training: TrainingConfig,
) -> DatasetSplit:
    """Create the notebook's deterministic, stratified 80/20 split."""
    if stratify_column not in frame:
        raise ValueError(f"Cannot stratify because {stratify_column!r} is absent")
    if frame[stratify_column].isna().any():
        raise ValueError(f"Cannot stratify on {stratify_column!r} with missing values")
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        frame,
        test_size=training.test_size,
        stratify=frame[stratify_column],
        random_state=training.split_seed,
    )
    return DatasetSplit(train=train.reset_index(drop=True), test=test.reset_index(drop=True))


def load_context_csv(path: Path, key_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, [key_column, "response"], path)
    return frame
