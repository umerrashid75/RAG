"""
Cross-encoder re-ranking via Cohere Rerank v3 (§7.4).
Phase P5 — deferred. Stub only.
"""
from __future__ import annotations

from app.models import RetrievedDoc


def rerank(
    query: str,
    docs: list[RetrievedDoc],
    *,
    top_k: int = 15,
    model: str = "rerank-english-v3.0",
) -> list[RetrievedDoc]:
    """P5: Re-score docs with Cohere cross-encoder; return top_k."""
    raise NotImplementedError("Cross-encoder reranking is implemented in Phase P5.")
