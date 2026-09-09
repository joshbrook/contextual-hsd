# Changelog

```yaml
document: "CHANGELOG"
owner: "Joshua Brook"
last_updated: "2026-09-09"
```

---

## Project-management baseline

Established the documentation baseline for the Contextual HSD research artefact.

- Added a research-oriented README, PRD, and current-state architecture reference grounded in the notebook and paper.
- Recorded reproducibility constraints and follow-up questions, including the current notebook/paper Gemini-model mismatch.

## Reusable experiment modules

Refactored the notebook implementation into importable Python modules while preserving the research experiment surface.

- Added typed configuration, deterministic data splitting/training, reusable context strategies, structured MLP evaluation, and dependency-light unit tests.
- Replaced shared-bucket latest-object retrieval with per-run Vertex batch manifests and isolated Cloud Storage prefixes.
- Reduced the notebook to Kaggle orchestration and reporting; reusable implementation lives in `src/`, one named experiment dispatcher replaces duplicate strategy wrappers, and the README/architecture record dependency, configuration, and artifact usage.
- Replaced the original notebook with `orchestrator.ipynb`, a guided starting point with one configuration block, selected-run preflight checks, and optional Vertex initialization.
- Added a context-source choice: use included generated-context data by default or submit and collect a new manifest-tracked Vertex batch without overwriting the checked-in CSVs.
- Made the included Latent Hatred and MAMI Base/Context CSVs the default notebook inputs; raw Latent Hatred stages remain an optional fallback.
- Added an early local-Python compatibility check so Python 3.14 does not try to build Kaggle-pinned NumPy from source.
- Added project-local virtual-environment and model-cache ignore rules for the documented local Python workflow.
- Made the notebook bootstrap pip in uv-created environments and select CPU automatically when CUDA is unavailable.
- Moved Kaggle Hugging Face cache writes from the read-only attached dataset to `/kaggle/working/`.
