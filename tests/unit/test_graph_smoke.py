"""
End-to-end smoke test for the agentic CRAG + Self-RAG LangGraph loop.

Uses mock providers and a fake retriever — no Qdrant, Neo4j, or network calls — so it
verifies the wiring (route → retrieve → grade → generate → reflect → finalize) runs and
produces a cited answer. This guards the headline feature against silent regressions.
"""
import pytest

from app.agents.graph import build_graph
from app.llm.providers import MockLLMProvider
from app.models import Chunk, RetrievedDoc


class _FakeRetriever:
    """Returns a fixed set of documents regardless of the query."""

    def __init__(self, docs: list[RetrievedDoc]) -> None:
        self._docs = docs

    def retrieve(self, query: str, *, top_k: int = 15) -> list[RetrievedDoc]:
        return self._docs[:top_k]


def _doc(doc_id: str, content: str) -> RetrievedDoc:
    chunk = Chunk(
        chunk_id=f"c_{doc_id}",
        parent_id=None,
        document_id=doc_id,
        document_type="Paper",
        content=content,
        metadata={"title": f"Paper {doc_id}"},
    )
    return RetrievedDoc(chunk=chunk, score=0.9, source="dense")


@pytest.mark.unit
def test_agentic_loop_produces_answer():
    retriever = _FakeRetriever(
        [
            _doc("2005.11401", "RAG combines a retriever with a generator."),
            _doc("1706.03762", "The Transformer relies on self-attention."),
        ]
    )
    llm = MockLLMProvider(response="RAG grounds generation in retrieved papers [2005.11401].")

    graph = build_graph(retriever, llm, max_iterations=2, crag_threshold=0.7)

    final_state = graph.invoke(
        {
            "query": "How does RAG reduce hallucination?",
            "retrieved": [],
            "crag_confidence": 0.0,
            "draft": "",
            "reflections": {},
            "iterations": 0,
            "answer": None,
        }
    )

    assert final_state["answer"], "agentic loop returned an empty answer"
    assert final_state["iterations"] >= 1
    assert final_state["iterations"] <= 2, "max_iterations guard was not respected"
    assert len(final_state["retrieved"]) >= 1
