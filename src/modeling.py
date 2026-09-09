"""PyTorch classifier training and result reporting."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import TrainingConfig


@dataclass(frozen=True)
class EvaluationResult:
    labels: tuple[str, ...]
    reports: tuple[dict[str, Any], ...]
    average_report: dict[str, dict[str, float]]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    best_true: np.ndarray
    best_predictions: np.ndarray


def prepare_targets(frame: pd.DataFrame, labels: str | Sequence[str], multilabel: bool) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...], object | None]:
    from sklearn.preprocessing import LabelEncoder

    columns = (labels,) if isinstance(labels, str) else tuple(labels)
    filtered = frame.copy()
    if multilabel:
        filtered = filtered[filtered["misogyny"] == "misogynous"].dropna(subset=list(columns))
        return filtered, filtered.loc[:, list(columns)].astype(float).to_numpy(), columns, None
    filtered = filtered.dropna(subset=[columns[0]])
    encoder = LabelEncoder().fit(filtered[columns[0]])
    return filtered, encoder.transform(filtered[columns[0]]), tuple(encoder.classes_), encoder


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(input_dimension: int, output_dimension: int, config: TrainingConfig):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dimension, config.hidden_dimension),
        nn.ReLU(),
        nn.Linear(config.hidden_dimension, config.hidden_dimension),
        nn.ReLU(),
        nn.Linear(config.hidden_dimension, config.hidden_dimension),
        nn.ReLU(),
        nn.Linear(config.hidden_dimension, output_dimension),
    )


def _average_reports(reports: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for report in reports:
        for label, values in report.items():
            if isinstance(values, dict):
                destination = aggregate.setdefault(label, {})
                for metric, value in values.items():
                    destination[metric] = destination.get(metric, 0.0) + float(value)
    for values in aggregate.values():
        for metric in values:
            values[metric] /= len(reports)
    return aggregate


def run_embedding_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: str | Sequence[str],
    training: TrainingConfig,
    multilabel: bool = False,
) -> EvaluationResult:
    """Train the original three-hidden-layer MLP over an embeddings dataframe column."""
    import torch
    import torch.optim as optim
    from sklearn.metrics import classification_report

    train_frame, train_targets, label_names, encoder = prepare_targets(train, labels, multilabel)
    if multilabel:
        test_frame, test_targets, _, _ = prepare_targets(test, labels, multilabel)
    else:
        label_column = labels if isinstance(labels, str) else labels[0]
        test_frame = test.dropna(subset=[label_column]).copy()
        if encoder is None:
            raise RuntimeError("A single-label experiment requires a fitted label encoder")
        try:
            test_targets = encoder.transform(test_frame[label_column])
        except ValueError as exc:
            raise ValueError("Test labels include a class absent from the training split") from exc
    if train_frame.empty or test_frame.empty:
        raise ValueError("No examples remain after target preparation")
    device = training.device if training.device == "cpu" or torch.cuda.is_available() else "cpu"
    x_train = torch.tensor(np.vstack(train_frame["embeddings"].to_numpy()), dtype=torch.float32, device=device)
    x_test = torch.tensor(np.vstack(test_frame["embeddings"].to_numpy()), dtype=torch.float32, device=device)
    y_train = torch.tensor(train_targets, dtype=torch.float32 if multilabel else torch.long, device=device)
    y_test = torch.tensor(test_targets, dtype=torch.float32 if multilabel else torch.long, device=device)

    reports: list[dict[str, Any]] = []
    best_f1, best_true, best_predictions = -1.0, np.array([]), np.array([])
    for run_index in range(training.runs):
        _seed_everything(training.seed_for_run(run_index))
        model = _build_model(x_train.shape[1], len(label_names), training).to(device)
        optimizer = optim.Adam(model.parameters(), lr=training.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()
        for _ in range(training.epochs):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(x_train), y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x_test)
            if multilabel:
                predictions = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                truth = y_test.cpu().numpy()
            else:
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                truth = y_test.cpu().numpy()
        report = classification_report(truth, predictions, target_names=label_names, output_dict=True, zero_division=0)
        reports.append(report)
        mean_f1 = float(np.mean([report[label]["f1-score"] for label in label_names]))
        if mean_f1 > best_f1:
            best_f1, best_true, best_predictions = mean_f1, truth, predictions

    average = _average_reports(reports)
    return EvaluationResult(
        labels=label_names,
        reports=tuple(reports),
        average_report=average,
        macro_precision=float(np.mean([average[label]["precision"] for label in label_names])),
        macro_recall=float(np.mean([average[label]["recall"] for label in label_names])),
        macro_f1=float(np.mean([average[label]["f1-score"] for label in label_names])),
        best_true=best_true,
        best_predictions=best_predictions,
    )


def format_evaluation(result: EvaluationResult) -> str:
    lines = ["Average classification report"]
    for label in result.labels:
        metrics = result.average_report[label]
        lines.append(f"{label:20s} P: {metrics['precision']:.2f} R: {metrics['recall']:.2f} F1: {metrics['f1-score']:.2f}")
    lines.append(f"Macro average       P: {result.macro_precision:.2f} R: {result.macro_recall:.2f} F1: {result.macro_f1:.2f}")
    return "\n".join(lines)
