"""
Tests for core Pydantic models and Protocol compliance.
"""
import pytest

from app.models import (
    Chunk,
    EmbeddingProvider,
    ExtractionResult,
    LLMProvider,
)


@pytest.mark.unit
class TestChunk:
    def test_parent_chunk_no_parent_id(self):
        chunk = Chunk(
            chunk_id="p1",
            parent_id=None,
            document_id="2310.06825",
            document_type="Paper",
            content="Test content",
        )
        assert chunk.parent_id is None

    def test_child_chunk_has_parent_id(self):
        chunk = Chunk(
            chunk_id="c1",
            parent_id="p1",
            document_id="2310.06825",
            document_type="Survey",
            content="Section content",
        )
        assert chunk.parent_id == "p1"

    def test_invalid_document_type_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="c1",
                parent_id=None,
                document_id="doc1",
                document_type="Invoice",  # invalid
                content="...",
            )

    def test_metadata_defaults_to_empty_dict(self):
        chunk = Chunk(
            chunk_id="c1",
            parent_id=None,
            document_id="doc1",
            document_type="Benchmark",
            content="...",
        )
        assert chunk.metadata == {}


@pytest.mark.unit
class TestProtocolCompliance:
    def test_mock_embedding_provider_satisfies_protocol(self):
        from app.indexing.embeddings import MockEmbeddings
        provider = MockEmbeddings()
        assert isinstance(provider, EmbeddingProvider)

    def test_mock_llm_provider_satisfies_protocol(self):
        from app.llm.providers import MockLLMProvider
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_mock_embeddings_return_correct_shape(self):
        from app.indexing.embeddings import MockEmbeddings
        emb = MockEmbeddings(dimensions=4)
        result = emb.embed(["hello", "world"])
        assert len(result) == 2
        assert all(len(v) == 4 for v in result)

    def test_mock_llm_returns_string_without_schema(self):
        from app.llm.providers import MockLLMProvider
        llm = MockLLMProvider(response="test answer")
        result = llm.generate("prompt")
        assert result == "test answer"

    def test_mock_llm_returns_model_with_schema(self):
        from pydantic import BaseModel

        from app.llm.providers import MockLLMProvider

        class MyModel(BaseModel):
            answer: str

        llm = MockLLMProvider(response='{"answer": "yes"}')
        result = llm.generate("prompt", schema=MyModel)
        assert isinstance(result, MyModel)
        assert result.answer == "yes"


@pytest.mark.unit
class TestExtractionResult:
    def test_empty_extraction_valid(self):
        result = ExtractionResult(entities=[], relations=[])
        assert result.entities == []
        assert result.relations == []

    def test_entity_type_validation(self):
        from pydantic import ValidationError

        from app.models import Entity
        with pytest.raises(ValidationError):
            Entity(name="Test", type="Banana")  # invalid type
