# Contextual HSD

```yaml
document: "README"
owner: "Joshua Brook"
last_updated: "2026-09-09"
```

This repo is the experimental artefact for *Leveraging LLMs for Context-Aware Implicit Textual and Multimodal Hate Speech Detection*. It evaluates whether an LLM can generate useful background context for implicit hateful tweets and misogynous memes, and compares ways of incorporating that context into a lightweight downstream classifier. The reusable implementation lives in the repository-root `src/` module; the notebook is a thin Kaggle runner.

> Content warning: the paper, data, and notebook contain examples of hateful, racist, sexist, and otherwise offensive content for research purposes.

## What It Does

- Runs textual experiments on the Latent Hatred dataset and textualised multimodal experiments on the MAMI dataset through importable Python functions.
- Compares zero-context, REL/Wikipedia, ConceptNet, LLM-generated-context, and direct-LLM classification setups.
- Uses one `run_experiment(EXPERIMENT, ...)` entry point for all embedding-based experiment strategies.
- Creates deterministic train/test splits and repeated MLP runs from typed, environment-backed configuration.
- Isolates each Vertex AI batch request with a local manifest and a unique Cloud Storage output prefix.

## Who This Is For

- Researchers reviewing, reproducing, or extending the accompanying study.
- Maintainers who need to rerun the guided Kaggle orchestrator or inspect the generated-context data.

## Getting Started

### Prerequisites

- Kaggle with GPU access and Python 3.10 is the documented execution environment. Local runs require Python 3.10–3.12; Python 3.14 is not supported by the pinned NumPy/PyTorch wheels.
- The checked-in Latent Hatred and MAMI Base/Context CSVs are sufficient for the normal embedding experiments. ConceptNet and Google Cloud access are needed only for their corresponding experiment paths.

### Setup

1. Create a Kaggle notebook and attach this repository as a dataset.
2. Open `orchestrator.ipynb` and select **Run All**; its default uses `data/latent hatred/LH_Base.csv`.
3. Change only the `RUN` choices you need. Paths are optional overrides for external/raw data.
4. Read the notebook's preflight message if it reports a missing path; it names every required file for the selected run.

For a local Windows run, create or select a Python 3.10–3.12 environment before opening the notebook. For example, if Python 3.10 is installed: `py -3.10 -m venv .venv`, then activate `.venv\\Scripts\\Activate.ps1` and start Jupyter from that environment. The notebook now stops before package installation with this same guidance when it detects Python 3.14 or later.

The notebook keeps downloaded Hugging Face models in its ignored `.cache/huggingface/` directory locally and in `/kaggle/working/contextual-hsd-cache/` on Kaggle, rather than attempting to write inside the read-only attached dataset.

### Run

Run `orchestrator.ipynb` from top to bottom in Kaggle. Start with `dataset='latent'`, `experiment='zero-context'`, and `task='binary'`; this baseline needs no cloud credentials. Latent Hatred supports `task='binary'` and `task='multiclass'`; MAMI supports `task='binary'` and `task='multilabel'`. The notebook imports `src` once, prepares only the selected strategy's dependencies, and runs the shared orchestrator. There is intentionally no standalone CLI or package-install metadata.

The default `device='auto'` selects CUDA in Kaggle or on a CUDA-enabled local PyTorch install, and CPU otherwise. CPU runs are functional but substantially slower than the intended Kaggle GPU environment.

### Configuration

The `RUN` block in `orchestrator.ipynb` is the normal configuration surface. It contains the dataset, experiment, task, only the paths needed for that run, and—if Vertex is enabled—the non-secret project and bucket names. It sets supplied values as environment variables before building `ExperimentConfig`; use raw environment variables only when automating a run outside the notebook.

| Variable | Purpose |
|----------|---------|
| `HSD_REPO_ROOT`, `HSD_LATENT_BASE_CSV`, `HSD_MAMI_CSV`, `HSD_CONCEPTNET_PATH` | Repository, Base-CSV overrides, and ConceptNet location. |
| `HSD_LATENT_HATRED_DIR` | Optional raw Latent Hatred stage-file fallback when a Base CSV is unavailable. |
| `HSD_ARTIFACT_DIR` | Local destination for request JSONL, batch manifests, and downloaded results. |
| `HSD_GCP_PROJECT`, `HSD_GCS_BUCKET`, `HSD_VERTEX_LOCATION`, `HSD_GEMINI_MODEL` | Non-secret Vertex AI settings. |
| `HSD_GCS_INPUT_PREFIX`, `HSD_GCS_OUTPUT_PREFIX` | Isolated Cloud Storage request and result prefixes. |

Cloud credentials remain in Kaggle Secrets and are never read from environment configuration or committed files.

### Included context or fresh LLM calls

For `append-embed`, `embed-concat`, and `context-embed`, `context_source='included'` is the default and uses the checked-in `data/` CSVs. To regenerate context, choose `context_source='rerun'` and use the notebook's `vertex_action` sequence: `submit-context`, wait for the batch to succeed, then `collect-context` with the printed manifest path. Collection writes a new CSV under `artifacts/results/`; set its path as `context_csv` before running the experiment. The checked-in CSVs are never overwritten. MAMI regeneration additionally requires its source images at `gs://<gcs_bucket>/MAMI_images/`.

### Which experiment should I run?

| Goal | Dataset | Experiment | Extra requirement |
|------|---------|------------|-------------------|
| First successful run | `latent` | `zero-context` | Included Base CSV only. |
| Compare LLM-generated context | `latent` or `mami` | `embed-concat` | Default: included context CSV. Optional: regenerate through the manifest-based Vertex flow. |
| Compare entity linking | `latent` or `mami` | `rel` | Kaggle internet access for REL and Wikipedia. |
| Compare ConceptNet | `latent` or `mami` | `conceptnet` | Local Numberbatch vectors. |
| Generate new context or direct predictions | either | Vertex batch flow | Kaggle `GOOGLE_API_KEY` secret plus project and bucket in `RUN`. |

## Operations

### Research Outputs

- `data/latent_hatred/LH_Base.csv`: Base Latent Hatred dataset in usable format.
- `data/latent_hatred/LH_Context.csv`: Latent Hatred records paired with generated background context.
- `data/latent_hatred/LH_NER.csv`: Latent Hatred records paired with named-entity annotations (experimental).
- `data/mami/MAMI_Base.csv`: Base MAMI dataset in usable format (no raw images).
- `data/mami/MAMI_Context.csv`: MAMI records paired with generated background context.
- `data/mami/MAMI_REL.csv`: Latent Hatred records paired with entity-linked annotations from REL (experimental).


## Repo Map

- `orchestrator.ipynb`: guided Kaggle runner and the recommended starting point.
- `src/`: reusable data, representation, model, experiment, and Vertex batch modules.
- `tests/`: dependency-light unit tests that mock model and cloud work.
- `requirements.txt`: pinned Kaggle runtime dependencies.
- `paper.pdf`: accompanying research paper, including methods, results, limitations, and ethical considerations.
- `data/`: checked-in generated-context and annotation data artefacts.
- `docs/`: project intent, current architecture, and change history.

## Key Docs

- `docs/PRD.md`
- `docs/ARCHITECTURE.md`
- `docs/CHANGELOG.md`
