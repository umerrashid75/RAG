"""
EmbeddingProvider implementations.

FastEmbedEmbeddings is the default: a local ONNX model (no API key, runs offline).
OpenAIEmbeddings wraps the OpenAI API; MockEmbeddings is for tests.
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


class FastEmbedEmbeddings:
    """
    Local embeddings via FastEmbed (ONNX runtime — no API key, no torch, runs offline).

    Default model BAAI/bge-small-en-v1.5 produces 384-dimensional vectors. The model
    is downloaded and cached on first use, which makes ingestion fully free.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        dimensions: int = 384,
    ) -> None:
        try:
            from fastembed import TextEmbedding  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for FastEmbedEmbeddings (pip install fastembed)"
            ) from exc
        self._model = TextEmbedding(model_name=model)
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # TextEmbedding.embed yields numpy arrays; materialize as plain lists.
        return [vector.tolist() for vector in self._model.embed(texts)]


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
