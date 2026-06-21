"""
Agentic query router (§8). Routes queries to global / local / multi-hop paths.
Phase P3 — full implementation. P1 uses a stub that always routes to 'local'.
"""
from __future__ import annotations

from typing import Literal

from app.agents.state import AgentState


def route_query(state: AgentState) -> Literal["global", "local", "multi_hop"]:
    """
    Classify the query into a retrieval route.

    Heuristics:
    - Synthesis / survey-style phrasing → 'global' (community-level summarization)
    - Contains an arXiv id → 'local' (a specific paper)
    - Default → 'local'
    """
    query = state["query"].lower()

    broad_phrases = (
        "compare", "comparison", "overview", "landscape", "survey", "trends",
        "state of the art", "approaches to", "how has", "evolution of",
    )
    if any(p in query for p in broad_phrases):
        return "global"

    import re
    # arXiv id patterns: "2310.06825" or "arXiv:2310.06825"
    if re.search(r"\b\d{4}\.\d{4,5}\b", query) or "arxiv:" in query:
        return "local"

    return "local"
