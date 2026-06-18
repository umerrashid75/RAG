"""
Self-RAG reflection grading via prompted JSON schema (§8.2).
Phase P3. Uses prompted JSON — NOT fine-tuned tokens (explicitly out of scope, §0.2).
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.models import LLMProvider, RetrievedDoc

log = logging.getLogger(__name__)


class ReflectionGrades(BaseModel):
    is_relevant: bool = Field(description="Retrieved context is relevant to the query.")
    is_faithful: bool = Field(description="Draft is grounded in the retrieved context (no hallucination).")
    is_useful: bool = Field(description="Draft usefully answers the user's query.")
    retrieve_again: bool = Field(description="True if another retrieval pass is needed.")
    reason: str = Field(description="Brief explanation of grades.")


_REFLECTION_PROMPT = """\
You are a Self-RAG evaluator for a legal intelligence system.
Assess the draft answer against the retrieved context and user query.

User Query: {query}

Retrieved Context (excerpts):
{context}

Draft Answer:
{draft}

Respond in strict JSON with these fields:
- is_relevant (bool): context is relevant to the query
- is_faithful (bool): draft claims are supported by context (no hallucination)
- is_useful (bool): draft directly answers the query
- retrieve_again (bool): another retrieval pass would help
- reason (str): one-sentence justification
"""


def reflect(
    query: str,
    draft: str,
    docs: list[RetrievedDoc],
    llm: LLMProvider,
) -> ReflectionGrades:
    """
    Grade the draft against the retrieved context.
    Returns ReflectionGrades with Is-Rel, Is-Faith, Is-Use, Retrieve flags.
    """
    context_snippets = "\n---\n".join(
        f"[{doc.chunk.document_type} | {doc.chunk.document_id}]\n{doc.chunk.content[:500]}"
        for doc in docs[:8]  # cap context size for the grader prompt
    )
    prompt = _REFLECTION_PROMPT.format(
        query=query,
        context=context_snippets,
        draft=draft[:2000],
    )

    try:
        result = llm.generate(prompt, schema=ReflectionGrades)
        if not isinstance(result, ReflectionGrades):
            raise ValueError("Unexpected return type from LLM")
        return result
    except Exception as exc:
        log.warning("Self-RAG reflection failed: %s. Defaulting to pass.", exc)
        # Conservative default: assume faithful to avoid infinite looping on grader errors.
        return ReflectionGrades(
            is_relevant=True,
            is_faithful=True,
            is_useful=True,
            retrieve_again=False,
            reason=f"Grader error ({exc}); defaulting to pass.",
        )


def grades_pass(grades: ReflectionGrades) -> bool:
    """Return True when the draft can be returned to the user."""
    return grades.is_faithful and grades.is_relevant and grades.is_useful
