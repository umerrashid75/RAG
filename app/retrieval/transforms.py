"""
Query transformation strategies: HyDE, step-back prompting (§7.1).
Phase P5 — deferred. Stubs only.
"""
from __future__ import annotations


def hyde_transform(query: str, llm) -> str:
    """Deferred: generate a hypothetical ideal answer passage to use as the query vector."""
    raise NotImplementedError("HyDE query transform is implemented in Phase P5.")


def step_back_transform(query: str, llm) -> str:
    """P5: Generate a broader principle-level query for step-back retrieval."""
    raise NotImplementedError("Step-back query transform is implemented in Phase P5.")
