"""
Typed LangGraph state object (§2). Threads through the full agentic loop.
"""
from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict

from app.models import RetrievedDoc


class AgentState(TypedDict):
    query: str
    route: Literal["global", "local", "multi_hop"]
    retrieved: list[RetrievedDoc]
    crag_confidence: float
    draft: str
    # Is-Rel, Is-Faith, Is-Use, Retrieve flags (§8.2)
    reflections: dict[str, Any]
    # Bounded by config.max_iterations to guarantee termination.
    iterations: int
    answer: str | None
