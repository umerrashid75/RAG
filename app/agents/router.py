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

    Heuristics (P1 stub — proper LLM-based routing is P3):
    - Contains a citation or patent number → 'local'
    - Starts with broad/synthesis phrases → 'global'
    - Default → 'local'

    P3 will replace this with an LLM classifier.
    """
    query = state["query"].lower()

    broad_phrases = (
        "strategies", "overview", "common approaches", "all cases", "history of",
        "compare", "landscape",
    )
    if any(p in query for p in broad_phrases):
        return "global"

    import re
    # Citation patterns: "410 U.S. 113", "US-10293847-B2", patent numbers
    if re.search(r"\b\d+\s+U\.S\.\s+\d+\b", query) or re.search(r"\bUS-\d+", query):
        return "local"

    return "local"
