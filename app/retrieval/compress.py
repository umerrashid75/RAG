"""
LLMLingua context compression (§7.4).
Phase P5 — deferred. Stub only.
"""
from __future__ import annotations

from app.models import RetrievedDoc


def compress_context(
    docs: list[RetrievedDoc],
    *,
    target_ratio: float = 0.5,
) -> list[RetrievedDoc]:
    """P5: Remove redundant tokens while preserving named entities and dates."""
    raise NotImplementedError("LLMLingua context compression is implemented in Phase P5.")
