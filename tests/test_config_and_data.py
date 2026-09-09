from pathlib import Path

import pandas as pd

from src.config import ExperimentPaths, TrainingConfig
from src.data import create_splits, load_latent_hatred, load_mami


def test_paths_read_environment_and_create_only_artifact_directories(monkeypatch, tmp_path):
    monkeypatch.setenv("HSD_ARTIFACT_DIR", str(tmp_path / "outputs"))
    paths = ExperimentPaths.from_environment(tmp_path)

    paths.create_artifact_directories()

    assert paths.root == tmp_path.resolve()
    assert (tmp_path / "outputs" / "manifests").is_dir()
    assert (tmp_path / "outputs" / "requests").is_dir()
    assert (tmp_path / "outputs" / "results").is_dir()


def test_default_paths_prefer_checked_in_base_files(monkeypatch, tmp_path):
    for name in ("HSD_LATENT_BASE_CSV", "HSD_MAMI_CSV"):
        monkeypatch.delenv(name, raising=False)

    paths = ExperimentPaths.from_environment(tmp_path)

    assert paths.latent_base_csv == tmp_path / "data" / "latent hatred" / "LH_Base.csv"
    assert paths.mami_csv == tmp_path / "data" / "mami" / "MAMI_Base.csv"


def test_load_mami_builds_textualised_post_and_split_is_deterministic(tmp_path):
    source = tmp_path / "mami.csv"
    pd.DataFrame(
        {
            "meme id": ["1.jpg", "2.jpg", "3.jpg", "4.jpg"],
            "extracted text": ["a", "b", "c", "d"],
            "image description": ["one", "two", "three", "four"],
            "misogyny": ["misogynous", "not misogynous", "misogynous", "not misogynous"],
            "shaming": [1, 0, 0, 0],
            "stereotype": [0, 0, 1, 0],
            "objectification": [0, 0, 0, 0],
            "violence": [0, 0, 0, 0],
        }
    ).to_csv(source, index=False)
    paths = ExperimentPaths(
        root=tmp_path,
        latent_hatred_dir=tmp_path,
        mami_csv=source,
        conceptnet_path=tmp_path / "conceptnet.txt",
        latent_context_csv=tmp_path / "latent.csv",
        mami_context_csv=tmp_path / "mami-context.csv",
        artifact_dir=tmp_path / "artifacts",
    )

    frame = load_mami(paths)
    first = create_splits(frame, "misogyny", TrainingConfig(test_size=0.5))
    second = create_splits(frame, "misogyny", TrainingConfig(test_size=0.5))

    assert frame.loc[0, "post"] == "a one"
    assert first.train["meme id"].tolist() == second.train["meme id"].tolist()
    assert first.test["meme id"].tolist() == second.test["meme id"].tolist()


def test_training_seed_sequence_is_stable():
    config = TrainingConfig(split_seed=101)
    assert [config.seed_for_run(index) for index in range(3)] == [101, 102, 103]


def test_training_device_reads_the_notebook_override(monkeypatch):
    monkeypatch.setenv("HSD_DEVICE", "cpu")

    assert TrainingConfig().device == "cpu"


def test_load_latent_hatred_recreates_the_notebook_stage_merge(tmp_path):
    identifiers = ["1", "2"]
    stage_rows = [
        {"post": ["first", "second"], "class": ["implicit_hate", "not_hate"]},
        {"post": ["first", "second"], "implicit_class": ["irony", None]},
        {
            "post": ["first", "second"],
            "target": ["group", None],
            "implied_statement": ["claim", None],
            "extra_implicit_class": [None, None],
        },
    ]
    for stage, values in enumerate(stage_rows, start=1):
        pd.DataFrame({"ID": identifiers}).to_csv(tmp_path / f"implicit_hate_v1_stg{stage}.tsv", sep="\t", index=False)
        pd.DataFrame(values).to_csv(tmp_path / f"implicit_hate_v1_stg{stage}_posts.tsv", sep="\t", index=False)
    paths = ExperimentPaths(
        root=tmp_path,
        latent_hatred_dir=tmp_path,
        mami_csv=tmp_path / "mami.csv",
        conceptnet_path=tmp_path / "conceptnet.txt",
        latent_context_csv=tmp_path / "latent.csv",
        mami_context_csv=tmp_path / "mami-context.csv",
        artifact_dir=tmp_path / "artifacts",
    )

    frame = load_latent_hatred(paths)

    assert frame[["ID", "post", "class", "binary_class"]].to_dict("records") == [
        {"ID": "1", "post": "first", "class": "implicit_hate", "binary_class": "hate"},
        {"ID": "2", "post": "second", "class": "not_hate", "binary_class": "not_hate"},
    ]
    assert frame.loc[0, "implicit_class"] == "irony"
    assert pd.isna(frame.loc[1, "implicit_class"])


def test_load_latent_hatred_prefers_a_checked_in_base_csv(tmp_path):
    base = tmp_path / "LH_Base.csv"
    pd.DataFrame(
        {
            "ID": ["1", "2"],
            "post": ["first", "second"],
            "class": ["implicit_hate", "not_hate"],
            "target": ["group", None],
            "implied_statement": ["claim", None],
            "implicit_class": ["irony", None],
            "extra_implicit_class": [None, None],
        }
    ).to_csv(base, index=False)
    paths = ExperimentPaths(
        root=tmp_path,
        latent_hatred_dir=tmp_path / "missing-raw-stages",
        mami_csv=tmp_path / "mami.csv",
        conceptnet_path=tmp_path / "conceptnet.txt",
        latent_context_csv=tmp_path / "latent.csv",
        mami_context_csv=tmp_path / "mami-context.csv",
        artifact_dir=tmp_path / "artifacts",
        latent_base_csv=base,
    )

    frame = load_latent_hatred(paths)

    assert frame[["ID", "post", "binary_class"]].to_dict("records") == [
        {"ID": 1, "post": "first", "binary_class": "hate"},
        {"ID": 2, "post": "second", "binary_class": "not_hate"},
    ]
