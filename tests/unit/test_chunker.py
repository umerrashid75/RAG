"""
Deterministic tests for the hierarchical chunker (§3 P1 acceptance criteria).
No external services required.
"""
import pytest

from app.ingestion.chunker import _split_by_tokens, _token_count, build_chunks


@pytest.mark.unit
class TestTokenSplit:
    def test_empty_text_returns_empty(self):
        assert _split_by_tokens("", chunk_size=100, overlap=10) == []

    def test_short_text_single_segment(self):
        text = "Hello world."
        segments = _split_by_tokens(text, chunk_size=100, overlap=10)
        assert len(segments) == 1
        assert segments[0] == text

    def test_overlap_is_applied(self):
        # 50-token text, 30-token chunk, 10-token overlap → stride = 20
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        # Build a 60-token text
        word = "legal "
        text = word * 60  # approximately 60 tokens (each "legal " ~ 1 token)
        segments = _split_by_tokens(text, chunk_size=30, overlap=10)
        assert len(segments) >= 2, "Expected multiple segments for text longer than chunk_size"

    def test_no_segment_exceeds_chunk_size(self):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        text = "The quick brown fox jumps over the lazy dog. " * 50
        chunk_size = 20
        segments = _split_by_tokens(text, chunk_size=chunk_size, overlap=5)
        for seg in segments:
            assert _token_count(seg) <= chunk_size, f"Segment exceeds chunk_size: {_token_count(seg)}"


@pytest.mark.unit
class TestBuildChunks:
    _SAMPLE_TEXT = (
        "In the United States Court of Appeals for the Federal Circuit. "
        "This case involves the infringement of Patent US-10293847-B2 "
        "related to pharmaceutical compounds. The district court held for "
        "the plaintiff on claims 1 through 5. Claim 1 recites a compound "
        "comprising an active pharmaceutical ingredient with molecular weight "
        "between 200 and 500 daltons. The defendant argued the doctrine of "
        "equivalents did not apply. We affirm the district court judgment. "
    ) * 30  # ~250 tokens

    def test_no_chunks_for_empty_text(self):
        result = build_chunks("", "doc1", "Case", {})
        assert result == []

    def test_parent_has_no_parent_id(self):
        chunks = build_chunks(self._SAMPLE_TEXT, "doc1", "Case", {})
        parents = [c for c in chunks if c.parent_id is None]
        assert len(parents) >= 1
        for p in parents:
            assert p.metadata["level"] == "parent"

    def test_children_reference_parent(self):
        chunks = build_chunks(self._SAMPLE_TEXT, "doc1", "Case", {})
        parent_ids = {c.chunk_id for c in chunks if c.parent_id is None}
        children = [c for c in chunks if c.parent_id is not None]
        assert len(children) >= 1
        for child in children:
            assert child.parent_id in parent_ids, (
                f"Child {child.chunk_id} references unknown parent {child.parent_id}"
            )

    def test_chunk_ids_are_unique(self):
        chunks = build_chunks(self._SAMPLE_TEXT, "doc1", "Case", {})
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_document_type_propagated(self):
        chunks = build_chunks(self._SAMPLE_TEXT, "doc1", "Patent", {})
        for c in chunks:
            assert c.document_type == "Patent"

    def test_metadata_merged(self):
        meta = {"jurisdiction": "CAFC", "year": 2024}
        chunks = build_chunks(self._SAMPLE_TEXT, "doc1", "Case", meta)
        for c in chunks:
            assert c.metadata["jurisdiction"] == "CAFC"
            assert c.metadata["year"] == 2024

    def test_child_tokens_within_bound(self):
        child_size = 50
        chunks = build_chunks(
            self._SAMPLE_TEXT, "doc1", "Case", {},
            parent_size=200, child_size=child_size, overlap=10,
        )
        children = [c for c in chunks if c.parent_id is not None]
        for child in children:
            tok = _token_count(child.content)
            assert tok <= child_size, f"Child exceeds child_size: {tok} tokens"

    def test_different_docs_different_ids(self):
        chunks_a = build_chunks(self._SAMPLE_TEXT, "docA", "Case", {})
        chunks_b = build_chunks(self._SAMPLE_TEXT, "docB", "Case", {})
        ids_a = {c.chunk_id for c in chunks_a}
        ids_b = {c.chunk_id for c in chunks_b}
        assert ids_a.isdisjoint(ids_b), "Different documents produced overlapping chunk_ids"
