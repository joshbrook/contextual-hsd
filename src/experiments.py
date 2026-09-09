"""Run the notebook's embedding experiments from one simple entry point."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .embeddings import append_context, concatenate_embeddings
from .modeling import EvaluationResult, run_embedding_experiment

EXPERIMENTS = (
    "zero-context",
    "rel",
    "conceptnet",
    "direct-llm",
    "append-embed",
    "embed-concat",
    "context-embed",
    "llm-enhance",
)


def run_experiment(
    experiment: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels,
    training,
    embedder: Callable[[Sequence[object]], Sequence[np.ndarray]] | None = None,
    context_provider: Callable[[object], object] | None = None,
    vector_provider: Callable[[Sequence[object]], Sequence[np.ndarray]] | None = None,
    context_embedder=None,
    predictions=None,
    text_column: str = "post",
    multilabel: bool = False,
) -> EvaluationResult | dict:
    """Run one named strategy using the same inputs as the original notebook.

    ``context_provider`` supplies REL or LLM context for the relevant strategies;
    for ``llm-enhance`` it returns the already enhanced text. ``vector_provider``
    supplies ConceptNet vectors. The caller controls expensive model/cloud work.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; choose one of {', '.join(EXPERIMENTS)}")
    if experiment == "direct-llm":
        if predictions is None:
            raise ValueError("direct-llm requires predictions")
        from sklearn.metrics import classification_report

        truth = test.loc[:, list(labels)].to_numpy() if multilabel else test[labels].tolist()
        return classification_report(truth, predictions, output_dict=True, zero_division=0)
    train = train.copy()
    test = test.copy()
    train_text = train[text_column].tolist()
    test_text = test[text_column].tolist()

    if experiment == "zero-context":
        if embedder is None:
            raise ValueError("zero-context requires an embedder")
        train["embeddings"] = list(embedder(train_text))
        test["embeddings"] = list(embedder(test_text))

    elif experiment in {"rel", "append-embed"}:
        if embedder is None or context_provider is None:
            raise ValueError(f"{experiment} requires an embedder and context_provider")
        train_context = [context_provider(text) for text in train_text]
        test_context = [context_provider(text) for text in test_text]
        train["embeddings"] = list(embedder(append_context(train_text, train_context)))
        test["embeddings"] = list(embedder(append_context(test_text, test_context)))

    elif experiment == "conceptnet":
        if embedder is None or vector_provider is None:
            raise ValueError("conceptnet requires an embedder and vector_provider")
        train["embeddings"] = concatenate_embeddings(embedder(train_text), vector_provider(train_text))
        test["embeddings"] = concatenate_embeddings(embedder(test_text), vector_provider(test_text))

    elif experiment == "embed-concat":
        if embedder is None or context_provider is None:
            raise ValueError("embed-concat requires an embedder and context_provider")
        train["embeddings"] = concatenate_embeddings(embedder(train_text), embedder([context_provider(text) for text in train_text]))
        test["embeddings"] = concatenate_embeddings(embedder(test_text), embedder([context_provider(text) for text in test_text]))

    elif experiment == "context-embed":
        if context_embedder is None or context_provider is None:
            raise ValueError("context-embed requires a context_embedder and context_provider")
        train["embeddings"] = context_embedder.embed_batch([context_provider(text) for text in train_text], train_text)
        test["embeddings"] = context_embedder.embed_batch([context_provider(text) for text in test_text], test_text)

    else:  # llm-enhance
        if embedder is None or context_provider is None:
            raise ValueError("llm-enhance requires an embedder and context_provider")
        train["embeddings"] = list(embedder([context_provider(text) for text in train_text]))
        test["embeddings"] = list(embedder([context_provider(text) for text in test_text]))

    return run_embedding_experiment(train, test, labels, training, multilabel)
