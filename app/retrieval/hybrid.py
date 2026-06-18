"""
Hybrid retrieval with Reciprocal Rank Fusion (§7.2).

RRF is implemented as a pure function so it can be unit-tested against
hand-computed results without touching Qdrant.

RRF_Score(d) = Σ_{m∈M} 1 / (k + r_m(d))
where M = {dense, sparse}, r_m(d) = rank of d in method m, k ≈ 60.
"""
from __future__ import annotations

import logging

from app.models import EmbeddingProvider, RetrievedDoc, Retriever

log = logging.getLogger(__name__)

_DEFAULT_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedDoc]],
    *,
    k: int = _DEFAULT_K,
) -> list[RetrievedDoc]:
    """
    Merge multiple ranked retrieval result lists with RRF.

    :param ranked_lists: Each inner list is already ordered by descending relevance
                         (rank 0 is the best). The source field on each RetrievedDoc
                         identifies which retriever produced it.
    :param k:            RRF constant (default 60, per spec §7.2).
    :returns:            A single de-duplicated list, sorted by descending RRF score.
                         Duplicates are detected by chunk_id; the first-seen RetrievedDoc
                         is kept and all hits are accumulated into its score.
    """
    scores: dict[str, float] = {}
    docs: dict[str, RetrievedDoc] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            cid = doc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in docs:
                docs[cid] = doc

    merged = sorted(docs.keys(), key=lambda cid: scores[cid], reverse=True)
    return [
        RetrievedDoc(
            chunk=docs[cid].chunk,
            score=scores[cid],
            source=docs[cid].source,
        )
        for cid in merged
    ]


class HybridRetriever:
    """
    Combines dense semantic search + BM25 keyword search, fused with RRF.
    Searches child chunks for precision; returns parent chunk content if available.
    """

    def __init__(
        self,
        vector_store,        # QdrantHybridStore
        embeddings: EmbeddingProvider,
        *,
        rrf_k: int = _DEFAULT_K,
    ) -> None:
        self._store = vector_store
        self._embeddings = embeddings
        self._rrf_k = rrf_k

    def retrieve(self, query: str, *, top_k: int = 15) -> list[RetrievedDoc]:
        """
        1. Embed the query.
        2. Run dense search.
        3. Run sparse/BM25 search (via Qdrant keyword search).
        4. Fuse with RRF.
        5. Return top_k fused results.
        """
        query_vector = self._embeddings.embed([query])[0]

        dense_results = self._store.search_dense(query_vector, top_k=top_k * 2)

        sparse_results = self._store.search_sparse(query, top_k=top_k * 2)

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self._rrf_k,
        )
        return fused[:top_k]
