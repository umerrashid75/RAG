"""
Graph-guided local search via Neo4j 1-2 hop traversal (§7.3).
Phase P2. Stub only in P1.
"""
from __future__ import annotations

from app.models import RetrievedDoc


def graph_local_search(
    query: str,
    graph_store,
    *,
    top_k: int = 10,
) -> list[RetrievedDoc]:
    """P2: Extract entities from query, traverse Neo4j 1-2 hops, return docs."""
    raise NotImplementedError("Graph local search is implemented in Phase P2.")
