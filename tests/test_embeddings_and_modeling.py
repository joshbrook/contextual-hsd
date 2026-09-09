import numpy as np
import pandas as pd
import pytest

from src.embeddings import append_context, concatenate_embeddings
from src.modeling import _average_reports, prepare_targets
from src.context import standardize_conceptnet_term


def test_append_and_concatenate_embeddings_preserve_row_alignment():
    assert append_context(["post"], ["context"]) == ["post [SEP] context"]
    values = concatenate_embeddings([np.array([1, 2])], [np.array([3])])
    assert values[0].tolist() == [1, 2, 3]


def test_prepare_targets_handles_single_and_multilabel_tasks():
    frame = pd.DataFrame(
        {
            "label": ["hate", "not_hate"],
            "misogyny": ["misogynous", "not misogynous"],
            "shaming": [1, 0],
            "violence": [0, 0],
        }
    )

    _, single_targets, labels, encoder = prepare_targets(frame, "label", multilabel=False)
    multilabel_frame, multi_targets, multi_labels, _ = prepare_targets(frame, ["shaming", "violence"], multilabel=True)

    assert labels == ("hate", "not_hate")
    assert encoder.inverse_transform(single_targets).tolist() == ["hate", "not_hate"]
    assert multilabel_frame.index.tolist() == [0]
    assert multi_labels == ("shaming", "violence")
    assert multi_targets.tolist() == [[1.0, 0.0]]


def test_average_reports_averages_metric_values():
    reports = [
        {"hate": {"precision": 0.2, "recall": 0.4, "f1-score": 0.3}},
        {"hate": {"precision": 0.6, "recall": 0.8, "f1-score": 0.7}},
    ]

    assert _average_reports(reports)["hate"] == pytest.approx({"precision": 0.4, "recall": 0.6, "f1-score": 0.5})


def test_conceptnet_term_standardizer_does_not_require_the_old_local_helper():
    assert standardize_conceptnet_term("Café noir!") == "cafe_noir"
