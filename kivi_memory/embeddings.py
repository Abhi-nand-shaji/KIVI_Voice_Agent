"""Embedding backends.

`OllamaEmbedder` is the real one (nomic-embed-text / bge-m3 running locally).
`HashEmbedder` is a deterministic, dependency-free stand-in so the prototype
still does vector retrieval when nothing is installed — it is a hashed
bag-of-tokens projection, not a semantic model, and it says so in `describe()`
rather than pretending otherwise.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .config import KiviConfig
from .core import cosine, tokens
from .llm import OllamaClient


class BaseEmbedder:
    name = "base"
    dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return cosine(a, b)

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "dim": self.dim, "semantic": False}


class HashEmbedder(BaseEmbedder):
    """Deterministic hashed bag-of-tokens with sublinear term weighting.

    Captures lexical overlap in vector form. It cannot capture paraphrase, so
    the engine reports `semantic: false` for it and the NLI layer is what
    carries meaning on the offline path.
    """

    name = "hash-bow"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts: dict[str, int] = {}
        for token in tokens(text):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector


class OllamaEmbedder(BaseEmbedder):
    name = "ollama"

    def __init__(self, client: OllamaClient, model: str, fallback: BaseEmbedder) -> None:
        self.client = client
        self.model = model
        self.fallback = fallback
        self.dim = 0
        self.degraded = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.client.embed(texts, model=self.model)
        if vectors is None:
            self.degraded = True
            return self.fallback.embed(texts)
        self.degraded = False
        if vectors:
            self.dim = len(vectors[0])
        return vectors

    def describe(self) -> dict[str, Any]:
        return {
            "backend": f"ollama:{self.model}" if not self.degraded else f"ollama:{self.model} (degraded -> hash-bow)",
            "dim": self.dim or self.fallback.dim,
            "semantic": not self.degraded,
        }


def get_embedder(config: KiviConfig, client: OllamaClient | None) -> BaseEmbedder:
    fallback = HashEmbedder()
    if config.retrieval == "lexical":
        return fallback
    if client is None:
        return fallback
    if config.retrieval == "hybrid":
        return OllamaEmbedder(client, config.embed_model, fallback)
    # auto
    if client.available() and client.has_model(config.embed_model):
        return OllamaEmbedder(client, config.embed_model, fallback)
    return fallback
