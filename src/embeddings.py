"""Embedding and context-fusion representations used by the experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import ModelConfig


def load_sentence_model(config: ModelConfig, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.sentence_model, device=device)


def embed_texts(model, texts: Iterable[object], device: str | None = None, batch_size: int = 64) -> list[np.ndarray]:
    """Encode text-like values, preserving the notebook's batch-size default."""
    values = [str(text) for text in texts]
    return list(model.encode(values, device=device, convert_to_numpy=True, batch_size=batch_size))


def append_context(posts: Sequence[object], contexts: Sequence[object], separator: str = " [SEP] ") -> list[str]:
    if len(posts) != len(contexts):
        raise ValueError("posts and contexts must have the same length")
    return [f"{post}{separator}{context}" for post, context in zip(posts, contexts)]


def concatenate_embeddings(*embedding_groups: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not embedding_groups:
        return []
    row_count = len(embedding_groups[0])
    if any(len(group) != row_count for group in embedding_groups):
        raise ValueError("All embedding groups must have the same number of rows")
    return [np.concatenate([np.asarray(group[index]) for group in embedding_groups]) for index in range(row_count)]


@dataclass
class ContextEmbedder:
    """Bourgeade-style hierarchical fusion with dependencies injected by the caller."""

    context_encoder: object
    message_encoder: object
    tokenizer: object
    device: str = "cuda"
    max_length: int = 256

    def embed(self, context: object, message: object) -> np.ndarray:
        import torch

        context_embedding = self.context_encoder.encode(str(context), convert_to_tensor=True, show_progress_bar=False)
        context_embedding = context_embedding.to(self.device).reshape(1, -1)
        encoding = self.tokenizer(
            str(message),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        message_embeddings = self.message_encoder.embeddings.word_embeddings(input_ids)
        _, _, embedding_dimension = message_embeddings.shape
        if context_embedding.size(1) != embedding_dimension:
            projection = torch.nn.Linear(context_embedding.size(1), embedding_dimension).to(self.device)
            context_embedding = projection(context_embedding)
        combined = torch.cat([context_embedding.unsqueeze(1), message_embeddings], dim=1)
        prefix_mask = torch.ones(input_ids.shape[0], 1, device=self.device)
        adjusted_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        self.message_encoder.eval()
        with torch.no_grad():
            output = self.message_encoder(input_ids=None, attention_mask=adjusted_mask, inputs_embeds=combined)
        return output.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

    def embed_batch(self, contexts: Sequence[object], messages: Sequence[object]) -> list[np.ndarray]:
        if len(contexts) != len(messages):
            raise ValueError("contexts and messages must have the same length")
        return [self.embed(context, message) for context, message in zip(contexts, messages)]


def load_context_embedder(config: ModelConfig, device: str = "cuda") -> ContextEmbedder:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    return ContextEmbedder(
        context_encoder=SentenceTransformer(config.sentence_model, device=device),
        message_encoder=AutoModel.from_pretrained(config.sentence_model).to(device),
        tokenizer=AutoTokenizer.from_pretrained(config.sentence_model),
        device=device,
        max_length=config.message_max_length,
    )
