import json

import pytest

from src.config import ExperimentPaths, VertexConfig
from src.vertex import (
    build_image_requests,
    build_text_requests,
    create_manifest,
    load_manifest,
    parse_image_result,
    parse_text_result,
    resolve_result_uri,
)


def _paths(tmp_path):
    return ExperimentPaths(
        root=tmp_path,
        latent_hatred_dir=tmp_path,
        mami_csv=tmp_path / "mami.csv",
        conceptnet_path=tmp_path / "conceptnet.txt",
        latent_context_csv=tmp_path / "latent.csv",
        mami_context_csv=tmp_path / "mami-context.csv",
        artifact_dir=tmp_path / "artifacts",
    )


def test_request_builders_and_manifest_use_a_unique_explicit_run_prefix(tmp_path):
    config = VertexConfig(bucket_name="research-bucket", model_id="test-model")
    text_requests = build_text_requests(["tweet"], "system ", "request: ")
    image_requests = build_image_requests(["1.jpg"], config, "system ", "request: ")
    manifest = create_manifest(_paths(tmp_path), config, "text-context", text_requests, run_id="run-123")

    assert text_requests[0]["request"]["contents"][0]["parts"][0]["text"] == "system request: tweet"
    assert image_requests[0]["request"]["contents"][0]["parts"][1]["fileData"]["fileUri"] == "gs://research-bucket/MAMI_images/1.jpg"
    assert manifest.input_uri == "gs://research-bucket/contextual-hsd/requests/run-123.jsonl"
    assert manifest.output_prefix_uri == "gs://research-bucket/contextual-hsd/predictions/run-123"
    assert load_manifest(tmp_path / "artifacts" / "manifests" / "run-123.json") == manifest


def test_result_parsers_ignore_rows_without_a_candidate(tmp_path):
    payload = [
        {
            "request": {"contents": [{"parts": [{"text": "prompt"}, {"fileData": {"fileUri": "gs://bucket/MAMI_images/1.jpg"}}]}]},
            "response": {"candidates": [{"content": {"parts": [{"text": "answer"}]}}]},
        },
        {"request": {"contents": []}, "response": {}},
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in payload), encoding="utf-8")

    assert parse_text_result(path).to_dict("records") == [{"post": "prompt", "response": "answer"}]
    assert parse_image_result(path).to_dict("records") == [{"meme id": "1.jpg", "response": "answer"}]


class _Blob:
    def __init__(self, name):
        self.name = name


class _Bucket:
    def __init__(self, names):
        self.names = names

    def list_blobs(self, prefix):
        return [_Blob(name) for name in self.names if name.startswith(prefix)]


class _Storage:
    def __init__(self, names):
        self.bucket_instance = _Bucket(names)

    def bucket(self, _name):
        return self.bucket_instance


def test_result_resolution_is_bound_to_the_manifest_prefix(tmp_path):
    config = VertexConfig(bucket_name="research-bucket")
    manifest = create_manifest(_paths(tmp_path), config, "text-context", [], run_id="isolated")
    storage = _Storage(
        [
            "contextual-hsd/predictions/isolated/result.jsonl",
            "contextual-hsd/predictions/other/result.jsonl",
        ]
    )

    resolved = resolve_result_uri(manifest, storage, config.bucket_name)

    assert resolved.result_uri == "gs://research-bucket/contextual-hsd/predictions/isolated/result.jsonl"


def test_result_resolution_rejects_ambiguous_or_missing_outputs(tmp_path):
    config = VertexConfig(bucket_name="research-bucket")
    manifest = create_manifest(_paths(tmp_path), config, "text-context", [], run_id="isolated")

    with pytest.raises(ValueError, match="exactly one JSONL"):
        resolve_result_uri(manifest, _Storage([]), config.bucket_name)
    with pytest.raises(ValueError, match="exactly one JSONL"):
        resolve_result_uri(
            manifest,
            _Storage(
                [
                    "contextual-hsd/predictions/isolated/first.jsonl",
                    "contextual-hsd/predictions/isolated/second.jsonl",
                ]
            ),
            config.bucket_name,
        )


def test_load_manifest_rejects_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid batch manifest"):
        load_manifest(path)
