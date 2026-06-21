"""
Tests for graph enrichment orchestration (no live Neo4j — uses a fake store).
"""
import pytest

from app.graph.enrich import enrich_graph
from app.models import Chunk, ExtractionResult


class _FakeGraphStore:
    def __init__(self) -> None:
        self.schema_ensured = False
        self.upserts: list[ExtractionResult] = []

    def ensure_schema(self) -> None:
        self.schema_ensured = True

    def upsert_extraction(self, result: ExtractionResult) -> None:
        self.upserts.append(result)


class _FakeLLM:
    """Returns a one-entity extraction regardless of input."""

    def generate(self, prompt: str, *, schema=None):
        return ExtractionResult.model_validate(
            {
                "entities": [{"name": "RAG", "type": "Method", "attributes": {}}],
                "relations": [],
            }
        )


def _parent(doc_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=f"p_{doc_id}",
        parent_id=None,
        document_id=doc_id,
        document_type="Paper",
        content=content,
    )


@pytest.mark.unit
def test_enrich_graph_extracts_per_document():
    store = _FakeGraphStore()
    chunks = [
        _parent("2005.11401", "Retrieval-augmented generation."),
        _parent("1706.03762", "Attention is all you need."),
    ]

    enriched = enrich_graph(chunks, _FakeLLM(), store)

    assert enriched == 2
    assert store.schema_ensured is True
    assert len(store.upserts) == 2


@pytest.mark.unit
def test_enrich_graph_skips_failed_document():
    class _FlakyLLM:
        def generate(self, prompt: str, *, schema=None):
            raise RuntimeError("extraction boom")

    store = _FakeGraphStore()
    enriched = enrich_graph([_parent("x", "text")], _FlakyLLM(), store)

    assert enriched == 0          # failure was swallowed
    assert store.upserts == []    # nothing upserted
