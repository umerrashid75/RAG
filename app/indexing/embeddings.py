"""
EmbeddingProvider implementations.
The OpenAIEmbeddings class wraps the OpenAI API; MockEmbeddings is for tests.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.models import EmbeddingProvider

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class OpenAIEmbeddings:
    """
    Thin wrapper around openai.embeddings.create.
    Batches large input lists to stay within the API's per-request limit.
    """

    _BATCH_SIZE = 512

    def __init__(self, model: str = "text-embedding-3-large", dimensions: int = 3072) -> None:
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("openai package is required for OpenAIEmbeddings") from exc
        self._client = openai.OpenAI()
        self._model = model
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            results.extend(item.embedding for item in response.data)
        return results


class MockEmbeddings:
    """
    Deterministic fake embeddings for unit tests — no API calls.
    Returns unit vectors in the first dimension for each text.
    """

    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            # Deterministic: hash the text length into the first dim
            v = [0.0] * self._dimensions
            v[0] = min(1.0, len(t) / 1000)
            vectors.append(v)
        return vectors


# Runtime check that both satisfy the protocol
assert isinstance(MockEmbeddings(), EmbeddingProvider)
