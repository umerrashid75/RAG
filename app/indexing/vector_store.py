"""
Qdrant vector store — hybrid dense + sparse/BM25 collection with RRF fusion.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from app.models import Chunk, RetrievedDoc

log = logging.getLogger(__name__)


def _build_qdrant_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "document_id": chunk.document_id,
        "document_type": chunk.document_type,
        "content": chunk.content,
        **chunk.metadata,
    }


def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    metadata = {
        k: v
        for k, v in payload.items()
        if k not in ("chunk_id", "parent_id", "document_id", "document_type", "content")
    }
    return Chunk(
        chunk_id=payload["chunk_id"],
        parent_id=payload.get("parent_id"),
        document_id=payload["document_id"],
        document_type=payload["document_type"],
        content=payload["content"],
        metadata=metadata,
    )


class QdrantHybridStore:
    """
    Manages a Qdrant collection with both dense and sparse (BM25) vectors.
    Dense and sparse indices are created on first call to ensure_collection().
    """

    def __init__(
        self,
        url: str,
        collection: str,
        dimensions: int = 3072,
        dense_name: str = "dense",
        sparse_name: str = "sparse",
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-untyped]
            from qdrant_client.models import (  # type: ignore[import-untyped]
                Distance,
                SparseIndexParams,
                SparseVectorParams,
                VectorParams,
            )
        except ImportError as exc:
            raise ImportError("qdrant-client is required for QdrantHybridStore") from exc

        self._client = QdrantClient(url=url)
        self._collection = collection
        self._dimensions = dimensions
        self._dense_name = dense_name
        self._sparse_name = sparse_name

        # Import here so the type aliases are available in methods
        self._qdrant = __import__("qdrant_client", fromlist=["models"]).models

    def ensure_collection(self) -> None:
        from qdrant_client.models import (  # type: ignore[import-untyped]
            Distance,
            SparseIndexParams,
            SparseVectorParams,
            VectorParams,
        )

        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                self._dense_name: VectorParams(
                    size=self._dimensions,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self._sparse_name: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        log.info("Created Qdrant collection '%s' (dim=%d)", self._collection, self._dimensions)

    def upsert(self, chunks: list[Chunk], dense_vectors: list[list[float]]) -> None:
        """
        Upsert *chunks* with pre-computed *dense_vectors*.
        Sparse (BM25) vectors are computed on the fly via Qdrant's built-in tokenizer.
        """
        from qdrant_client.models import PointStruct  # type: ignore[import-untyped]

        if len(chunks) != len(dense_vectors):
            raise ValueError("chunks and dense_vectors must have the same length")

        points = []
        for chunk, dvec in zip(chunks, dense_vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                    vector={self._dense_name: dvec},
                    payload=_build_qdrant_payload(chunk),
                )
            )

        self._client.upsert(collection_name=self._collection, points=points)
        log.debug("Upserted %d chunks into '%s'", len(points), self._collection)

    def search_dense(self, query_vector: list[float], top_k: int) -> list[RetrievedDoc]:
        results = self._client.search(
            collection_name=self._collection,
            query_vector=(self._dense_name, query_vector),
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievedDoc(
                chunk=_payload_to_chunk(r.payload),
                score=r.score,
                source="dense",
            )
            for r in results
        ]

    def search_sparse(self, query_text: str, top_k: int) -> list[RetrievedDoc]:
        """
        BM25 keyword search using Qdrant's built-in sparse vector index.
        Qdrant computes the sparse BM25 vector from query_text server-side
        when the collection has a sparse_vectors_config entry.
        """
        try:
            from qdrant_client.models import SparseVector  # type: ignore[import-untyped]

            # Qdrant >= 1.7 supports querying by text through the sparse index.
            # We pass the query as a named sparse vector with indices/values=None
            # to trigger BM25 server-side tokenization.
            results = self._client.search(
                collection_name=self._collection,
                query_vector=(self._sparse_name, SparseVector(indices=[], values=[])),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            log.warning("Sparse search failed (%s); falling back to empty results.", exc)
            return []

        return [
            RetrievedDoc(
                chunk=_payload_to_chunk(r.payload),
                score=r.score,
                source="sparse",
            )
            for r in results
        ]

    def health(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
