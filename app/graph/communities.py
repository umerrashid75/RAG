"""
Leiden community detection + summarization via Neo4j GDS (§6.3, P4).
Phase P4 — deferred. Stub only.
"""
from __future__ import annotations


def detect_communities(graph_store, *, resolution: float = 0.85, max_levels: int = 3) -> dict:
    """P4: Run Leiden via Neo4j GDS; return community assignments."""
    raise NotImplementedError("Leiden community detection is implemented in Phase P4.")


def summarize_community(community_nodes: list[dict], llm) -> str:
    """P4: Summarize a Leiden community into a searchable report."""
    raise NotImplementedError("Community summarization is implemented in Phase P4.")
