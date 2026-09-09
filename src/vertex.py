"""Explicit Vertex batch workflows with per-run request/output manifests."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .config import ExperimentPaths, VertexConfig


@dataclass(frozen=True)
class BatchManifest:
    run_id: str
    kind: str
    model_id: str
    input_uri: str
    output_prefix_uri: str
    local_request_path: str
    local_manifest_path: str
    created_at: str
    job_name: str | None = None
    result_uri: str | None = None


def _uri(bucket_name: str, path: str) -> str:
    return f"gs://{bucket_name}/{path.lstrip('/')}"


def _write_jsonl(path: Path, requests: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")


def _write_manifest(manifest: BatchManifest) -> None:
    path = Path(manifest.local_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> BatchManifest:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        return BatchManifest(**values)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid batch manifest: {path}") from exc


def build_text_requests(texts: Iterable[object], system: str, request: str) -> list[dict]:
    return [
        {"request": {"contents": [{"role": "user", "parts": [{"text": f"{system}{request}{text}"}]}]}}
        for text in texts
    ]


def build_image_requests(image_ids: Iterable[object], config: VertexConfig, system: str, request: str) -> list[dict]:
    return [
        {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": f"{system}{request}"},
                            {
                                "fileData": {
                                    "fileUri": _uri(config.bucket_name, f"MAMI_images/{image_id}"),
                                    "mimeType": "image/jpeg",
                                }
                            },
                        ],
                    }
                ]
            }
        }
        for image_id in image_ids
    ]


def create_manifest(
    paths: ExperimentPaths,
    config: VertexConfig,
    kind: str,
    requests: Sequence[dict],
    run_id: str | None = None,
) -> BatchManifest:
    """Write a local request file and manifest before any cloud side effect."""
    paths.create_artifact_directories()
    identifier = run_id or uuid.uuid4().hex
    request_path = paths.artifact_dir / "requests" / f"{identifier}.jsonl"
    manifest_path = paths.artifact_dir / "manifests" / f"{identifier}.json"
    input_uri = _uri(config.bucket_name, f"{config.input_prefix}/{identifier}.jsonl")
    output_prefix_uri = _uri(config.bucket_name, f"{config.output_prefix}/{identifier}")
    _write_jsonl(request_path, requests)
    manifest = BatchManifest(
        run_id=identifier,
        kind=kind,
        model_id=config.model_id,
        input_uri=input_uri,
        output_prefix_uri=output_prefix_uri,
        local_request_path=str(request_path),
        local_manifest_path=str(manifest_path),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_manifest(manifest)
    return manifest


def submit_batch(manifest: BatchManifest, vertex_config: VertexConfig, storage_client, genai_client, batch_config_factory=None) -> BatchManifest:
    """Upload the manifest's exact request file and submit one isolated batch job."""
    bucket = storage_client.bucket(vertex_config.bucket_name)
    input_path = manifest.input_uri.removeprefix(f"gs://{vertex_config.bucket_name}/")
    bucket.blob(input_path).upload_from_filename(manifest.local_request_path)
    if batch_config_factory is None:
        from google.genai.types import CreateBatchJobConfig

        batch_config_factory = CreateBatchJobConfig
    job = genai_client.batches.create(
        model=manifest.model_id,
        src=manifest.input_uri,
        config=batch_config_factory(dest=manifest.output_prefix_uri),
    )
    submitted = replace(manifest, job_name=job.name)
    _write_manifest(submitted)
    return submitted


def batch_status(manifest: BatchManifest, genai_client) -> str:
    if not manifest.job_name:
        raise ValueError("Cannot inspect a batch that has not been submitted")
    return str(genai_client.batches.get(name=manifest.job_name).state)


def resolve_result_uri(manifest: BatchManifest, storage_client, bucket_name: str) -> BatchManifest:
    """Resolve exactly one JSONL output under this run's isolated prefix."""
    if manifest.result_uri:
        return manifest
    prefix = manifest.output_prefix_uri.removeprefix(f"gs://{bucket_name}/")
    candidates = [
        blob.name
        for blob in storage_client.bucket(bucket_name).list_blobs(prefix=prefix)
        if blob.name.endswith(".jsonl")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one JSONL result for run {manifest.run_id}, found {len(candidates)}")
    resolved = replace(manifest, result_uri=_uri(bucket_name, candidates[0]))
    _write_manifest(resolved)
    return resolved


def _download_result(manifest: BatchManifest, paths: ExperimentPaths, storage_client, bucket_name: str) -> Path:
    if not manifest.result_uri:
        raise ValueError("Resolve the batch result before downloading it")
    result_path = paths.artifact_dir / "results" / f"{manifest.run_id}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    blob_name = manifest.result_uri.removeprefix(f"gs://{bucket_name}/")
    storage_client.bucket(bucket_name).blob(blob_name).download_to_filename(str(result_path))
    return result_path


def parse_text_result(path: Path) -> pd.DataFrame:
    rows = pd.read_json(path, lines=True)
    posts, responses = [], []
    for request, response in zip(rows.get("request", []), rows.get("response", [])):
        try:
            content = response["candidates"][0]["content"]["parts"][0]["text"]
            post = request["contents"][0]["parts"][0]["text"]
        except (IndexError, KeyError, TypeError):
            continue
        posts.append(post)
        responses.append(content)
    return pd.DataFrame({"post": posts, "response": responses})


def parse_image_result(path: Path) -> pd.DataFrame:
    rows = pd.read_json(path, lines=True)
    images, responses = [], []
    for request, response in zip(rows.get("request", []), rows.get("response", [])):
        try:
            content = response["candidates"][0]["content"]["parts"][0]["text"]
            image_uri = request["contents"][0]["parts"][1]["fileData"]["fileUri"]
        except (IndexError, KeyError, TypeError):
            continue
        images.append(image_uri.rsplit("/", 1)[-1])
        responses.append(content)
    return pd.DataFrame({"meme id": images, "response": responses})


def submit_requests(
    kind: str,
    requests: Sequence[dict],
    paths: ExperimentPaths,
    config: VertexConfig,
    storage_client,
    genai_client,
    run_id: str | None = None,
) -> BatchManifest:
    """Write, upload, and submit one batch request set."""
    return submit_batch(create_manifest(paths, config, kind, requests, run_id), config, storage_client, genai_client)


def collect_result(
    manifest: BatchManifest,
    paths: ExperimentPaths,
    config: VertexConfig,
    storage_client,
    kind: str = "text",
) -> tuple[BatchManifest, pd.DataFrame]:
    """Download and parse a completed text or image batch."""
    if kind not in {"text", "image"}:
        raise ValueError("kind must be 'text' or 'image'")
    manifest = resolve_result_uri(manifest, storage_client, config.bucket_name)
    result_path = _download_result(manifest, paths, storage_client, config.bucket_name)
    return manifest, (parse_image_result(result_path) if kind == "image" else parse_text_result(result_path))
