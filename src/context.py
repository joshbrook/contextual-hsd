"""External knowledge baselines: REL/Wikipedia and ConceptNet Numberbatch."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable
import unicodedata

import numpy as np

REL_API_URL = "https://rel.cs.ru.nl/api"


@dataclass
class RelWikipediaContextProvider:
    """Retrieve concise Wikipedia descriptions for high-confidence REL entities."""

    wiki: object
    post: Callable[..., object]
    api_url: str = REL_API_URL

    @classmethod
    def create_default(cls) -> "RelWikipediaContextProvider":
        import requests
        import wikipediaapi

        return cls(
            wiki=wikipediaapi.Wikipedia(user_agent="NLPContextExtraction", language="en"),
            post=requests.post,
        )

    def extract(self, text: object) -> str:
        try:
            response = self.post(self.api_url, json={"text": str(text)}, timeout=30)
            response.raise_for_status()
            mentions = response.json()
            names = [mention[3] for mention in mentions if mention[4] > 0.2 and mention[5] > 0.4]
            descriptions = [self._description(name) for name in names]
            return " ".join(description for description in descriptions if description)
        except Exception:
            return ""

    def _description(self, entity: str) -> str:
        page = self.wiki.page(entity)
        if not page.exists():
            return ""
        sentences = page.summary.split(". ")
        return ". ".join(sentences[:2]).strip()


def load_conceptnet_embeddings(path):
    from gensim.models import KeyedVectors

    return KeyedVectors.load_word2vec_format(str(path), binary=False)


def standardize_conceptnet_term(term: object) -> str:
    """Return the Numberbatch vocabulary key for an English unigram or n-gram.

    Older Kaggle sessions may provide the notebook's local ``numberbatch_uri``
    helper. The fallback keeps this repository runnable without that unpublished
    dependency and covers the normalized keys used by Numberbatch text files.
    """
    try:
        from numberbatch_uri import standardized_uri
    except ImportError:
        normalized = unicodedata.normalize("NFKD", str(term)).encode("ascii", "ignore").decode("ascii").lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
        return normalized
    return standardized_uri("en", str(term)).split("/")[-1]


def conceptnet_entities(text: object, embeddings) -> list[str]:
    import nltk

    words = nltk.word_tokenize(str(text))
    grams = [[word] for word in words]
    grams += list(nltk.bigrams(words)) if len(words) >= 2 else []
    grams += list(nltk.trigrams(words)) if len(words) >= 3 else []
    entities = {
        standardize_conceptnet_term(" ".join(gram))
        for gram in grams
    }
    return sorted(entity for entity in entities if entity in embeddings)


def average_and_normalize(vectors: Iterable[np.ndarray], dimension: int = 300) -> np.ndarray:
    values = [np.asarray(vector) for vector in vectors]
    if not values:
        return np.zeros(dimension)
    average = np.mean(values, axis=0)
    norm = np.linalg.norm(average)
    return average / norm if norm else average


def conceptnet_context_vectors(texts: Iterable[object], embeddings, dimension: int = 300) -> list[np.ndarray]:
    vectors = []
    for text in texts:
        entities = conceptnet_entities(text, embeddings)
        vectors.append(average_and_normalize([embeddings[entity] for entity in entities], dimension))
    return vectors
