"""
Embedding utilities: cosine similarity, a caching embedder, and anchor sets.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

RawEmbedFn = Callable[[list[str]], list[list[float]]]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, robust to zero vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Embedder:
    """Wraps a raw embed function with a simple in-memory cache.

    Anchor phrases and repeated user inputs are embedded at most once, which
    matters because every embedding call is a billed network round-trip.
    """

    def __init__(self, raw_embed: RawEmbedFn):
        self._raw = raw_embed
        self._cache: dict[str, np.ndarray] = {}

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        # Only send the strings we have not already cached.
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vec in zip(missing, self._raw(missing)):
                self._cache[text] = np.asarray(vec, dtype=np.float32)
        return [self._cache[t] for t in texts]

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]


class AnchorSet:
    """A labelled group of reference phrases with precomputed embeddings.

    ``max_similarity`` returns the cosine similarity of an input vector to its
    *nearest* anchor. Max (rather than mean) is used because a topic is a
    union of sub-areas -- a question about aphids should match the "pests"
    anchor strongly even if it is unlike the "soil" anchor.
    """

    def __init__(self, label: str, phrases: list[str], embedder: Embedder):
        self.label = label
        self.phrases = phrases
        self.vectors = embedder.embed_many(phrases)

    def max_similarity(self, vec: np.ndarray) -> float:
        return max((cosine(vec, a) for a in self.vectors), default=0.0)
