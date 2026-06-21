"""
Corrective RAG (CRAG) — retrieval grader + Tavily web fallback (§8.1).
Phase P3.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel

from app.models import LLMProvider, RetrievedDoc

log = logging.getLogger(__name__)


class RelevanceGrade(BaseModel):
    score: float    # 0.0 – 1.0
    reason: str


_GRADE_PROMPT = """\
You are a research relevance grader. Given a user question and a retrieved chunk from
a research paper, score the chunk's relevance to answering the question.

Query: {query}

Chunk:
{content}

Respond in JSON: {{"score": <0.0-1.0>, "reason": "<brief explanation>"}}
"""


def grade_documents(
    query: str,
    docs: list[RetrievedDoc],
    llm: LLMProvider,
) -> tuple[list[RetrievedDoc], float]:
    """
    Grade each retrieved doc for relevance.
    Returns (relevant_docs, confidence_score).
    confidence_score = fraction of docs rated >= 0.5.
    """
    if not docs:
        return [], 0.0

    grades: list[float] = []
    relevant: list[RetrievedDoc] = []

    for doc in docs:
        prompt = _GRADE_PROMPT.format(query=query, content=doc.chunk.content[:800])
        try:
            grade = llm.generate(prompt, schema=RelevanceGrade)
            if not isinstance(grade, RelevanceGrade):
                grade = RelevanceGrade(score=0.5, reason="parse failure")
        except Exception as exc:
            log.warning("Grading failed for chunk %s: %s", doc.chunk.chunk_id, exc)
            grade = RelevanceGrade(score=0.5, reason="error")

        grades.append(grade.score)
        if grade.score >= 0.5:
            relevant.append(doc)

    confidence = sum(1 for g in grades if g >= 0.5) / len(grades)
    return relevant, confidence


def web_fallback(query: str, tavily_api_key: str, max_results: int = 5) -> list[RetrievedDoc]:
    """Tavily web search fallback when CRAG confidence is below threshold."""
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("tavily-python is required for web_fallback") from exc

    from app.models import Chunk

    client = TavilyClient(api_key=tavily_api_key)
    results = client.search(query=query, max_results=max_results)

    docs: list[RetrievedDoc] = []
    for i, r in enumerate(results.get("results", [])):
        chunk = Chunk(
            chunk_id=f"web_{i}",
            parent_id=None,
            document_id=r.get("url", f"web_{i}"),
            document_type="Paper",  # best-effort default for web-sourced material
            content=r.get("content", ""),
            metadata={"url": r.get("url", ""), "title": r.get("title", "")},
        )
        docs.append(RetrievedDoc(chunk=chunk, score=r.get("score", 0.5), source="web"))

    return docs
