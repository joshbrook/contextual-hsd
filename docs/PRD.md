# Product Requirements Document

```yaml
document: "PRD"
project: "Contextual HSD"
owner: "Joshua Wolfe Brook"
status: "Active"
last_updated: "2026-09-08"
```

---

## 1. What It Is

Contextual HSD is a research project and reproducibility artefact investigating LLM-generated background context for implicit textual and multimodal hate-speech detection (HSD). It evaluates context-generation and context-incorporation strategies using the Latent Hatred tweet dataset and the MAMI misogynous-meme dataset.

## 2. Why It Exists

Implicit hate and hateful memes can depend on cultural references, irony, coded language, and image-text interaction that are not apparent in the raw text alone. Earlier context-aware approaches based on entity linking can miss relevant information or introduce noisy knowledge. This project tests whether post-specific LLM-generated background context is a more useful input to a lightweight supervised classifier.

## 3. Who Uses It

| Role | What they need from this |
|------|--------------------------|
| Research author or maintainer | A documented experimental record that supports inspecting and rerunning the reported comparisons. |
| Reproduction researcher | The paper, notebook, generated-context data, and clear account of missing external dependencies. |
| HSD researcher | A modular basis for testing alternative context generators or incorporation methods while retaining comparable baselines. |

## 4. How It Works

1. Prepare the Latent Hatred and MAMI datasets and make a stratified 80/20 train/test split.
2. Build zero-context and entity-based baselines, then generate neutral background context for each post or meme with an LLM.
3. Encode inputs with SBERT, apply one of several context-incorporation strategies, and train an MLP classifier.
4. Report macro F1 and task-specific F1 over repeated runs; compare with a direct LLM classifier and inspect error cases.

## 5. First Version Scope

The research artefact should:

- Preserve the notebook implementing the reported baseline and LLM-context experiments.
- Publish the accompanying paper and generated-context data needed to inspect the work.
- Support textual binary and implicit-class HSD on Latent Hatred and binary/multi-label misogyny detection on MAMI.
- Compare zero-context, REL/Wikipedia, ConceptNet, LLM-generated context, and direct-LLM classification configurations.

## 6. Out of Scope for Now

- Production content moderation or automated enforcement - this repository reports experimental research results only.
- A fully packaged local or cloud deployment - the current implementation is a Kaggle notebook with external inputs and credentials.
- Claims of universal or multilingual generalization - the evaluated datasets are English-language and limited in time and domain.
- Replacing human review, appeals, or policy governance - the paper identifies material ethical and classification risks.

## 7. Key Requirements

- Keep experimental comparisons traceable to a dataset, task, context strategy, and evaluation metric.
- Preserve a zero-context baseline and the two entity-based baselines when making comparable extensions.
- Prevent label leakage in generated context: context prompts must remain neutral and must not tell the LLM that its output will be used for hate-speech classification.
- Report macro F1 and relevant class-level measures across repeated runs rather than a single selected run.
- Provide a content warning when presenting material that contains hateful, racist, sexist, or otherwise offensive examples.

### Security and Privacy

- The notebook retrieves `GOOGLE_API_KEY` from Kaggle Secrets and uses Google Cloud services for batch LLM work; secrets must never be committed.
- The data and paper contain harmful content. Access, storage, exports, and displays should be limited to the research need.
- Open question: the repository has no documented retention, access-control, or provenance policy for cloud batch inputs and outputs.

## 8. Key Design Choices

- Use full-post LLM context generation rather than only entity-centric retrieval - implicit meaning and ambiguous references can depend on the whole post.
- Use an SBERT-plus-MLP downstream pipeline - it isolates the effect of generated context from a general-purpose LLM classifier.
- Compare four incorporation strategies - appending, embedding concatenation, hierarchical Context-Embed fusion, and LLM enhancement test different levels of separation between input and generated context.
- Include a direct LLM classifier baseline - it makes the trade-off between contextual augmentation and end-to-end LLM prediction visible.

## 9. Success Criteria

- Quality: reproduce the study's reported experimental setup and make the included generated-context outputs inspectable.
- Comparability: evaluate additions against the documented zero-context, REL, ConceptNet, and direct-LLM baselines on the same splits and metrics.
- Transparency: record model, prompt, data, runtime, and external-dependency changes that can affect results.
- Safety: avoid presenting experimental results as moderation-policy recommendations and maintain content warnings.

Measurement approach: use the paper's macro-F1 and per-class/positive-class F1 reporting approach, averaged across ten runs where the experiment remains comparable.

## 10. Key Questions and Risks

- Open question: how can the notebook be packaged with pinned dependencies, reproducible data-access instructions, and validated execution without exposing credentials or restricted assets?
- Open question: which alternative models, prompts, or decoding settings improve context quality without increasing semantic drift or label leakage?
- Open question: how should generated context itself be evaluated for factuality, relevance, and harmful over-interpretation, rather than only by downstream F1?
- Generated context can obscure explicit harm or introduce unsupported hateful associations, producing false negatives or false positives.
- Dataset labels, especially for implicit hate, can be subjective or ambiguous; English-language benchmark performance may not generalize.

## 11. Decisions

| Date | Decision | Reason |
|------|----------|--------|
| 2026-09-08 | Treat this repository as a research/reproducibility artefact rather than a production moderation system. | The current implementation and paper describe controlled experiments, not deployment. |
| 2026-09-08 | Document the notebook's current external dependencies and credentials as constraints instead of implying a self-contained local run. | The repository contains no lockfile, complete source datasets, image assets, or cloud configuration. |
