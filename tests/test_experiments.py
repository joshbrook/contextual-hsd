import numpy as np
import pandas as pd
import pytest

from src import experiments
from src.config import TrainingConfig


def _embed(values):
    return [np.array([len(str(value))], dtype=float) for value in values]


def _frames():
    train = pd.DataFrame({"post": ["a", "bb"], "label": ["x", "y"]})
    test = pd.DataFrame({"post": ["ccc", "d"], "label": ["x", "y"]})
    return train, test


def test_one_runner_handles_every_embedding_strategy(monkeypatch):
    captured = []

    def fake_evaluate(train, test, labels, training, multilabel):
        captured.append((train, test, labels, multilabel))
        return "evaluated"

    monkeypatch.setattr(experiments, "run_embedding_experiment", fake_evaluate)
    train, test = _frames()
    config = TrainingConfig(runs=1, epochs=1, device="cpu")
    context_embedder = type("Embedder", (), {"embed_batch": lambda self, contexts, messages: _embed(messages)})()

    for name in experiments.EXPERIMENTS:
        if name == "direct-llm":
            continue
        kwargs = {"embedder": _embed, "context_provider": lambda value: f"ctx:{value}", "vector_provider": _embed}
        if name == "context-embed":
            kwargs["context_embedder"] = context_embedder
        assert experiments.run_experiment(name, train, test, "label", config, **kwargs) == "evaluated"

    assert len(captured) == len(experiments.EXPERIMENTS) - 1
    assert all("embeddings" in train_frame for train_frame, _, _, _ in captured)


def test_runner_logs_strategy_and_embedding_summary(monkeypatch, caplog):
    monkeypatch.setattr(experiments, "run_embedding_experiment", lambda *args, **kwargs: "evaluated")
    train, test = _frames()

    with caplog.at_level("INFO", logger="src.experiments"):
        experiments.run_experiment("zero-context", train, test, "label", TrainingConfig(device="cpu"), embedder=_embed)

    messages = caplog.messages
    assert any("Starting experiment=zero-context" in message for message in messages)
    assert any("Prepared experiment=zero-context embeddings" in message for message in messages)


def test_runner_names_and_required_inputs_are_clear():
    train, test = _frames()
    config = TrainingConfig(runs=1, epochs=1, device="cpu")

    with pytest.raises(ValueError, match="Unknown experiment"):
        experiments.run_experiment("unknown", train, test, "label", config)
    with pytest.raises(ValueError, match="requires an embedder"):
        experiments.run_experiment("zero-context", train, test, "label", config)


def test_direct_llm_evaluation_uses_the_same_experiment_runner():
    train, test = _frames()
    report = experiments.run_experiment(
        "direct-llm",
        train,
        test,
        "label",
        TrainingConfig(runs=1, epochs=1, device="cpu"),
        predictions=["x", "x"],
    )

    assert report["accuracy"] == 0.5
