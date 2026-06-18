"""
Deterministic tests for RRF fusion math (§3 P1 acceptance criteria).
The spec requires RRF to be verified against hand-computed ranks.
All tests run without Qdrant or any external service.
"""
import pytest

from app.models import Chunk, RetrievedDoc
from app.retrieval.hybrid import reciprocal_rank_fusion


def _make_doc(chunk_id: str, source: str = "dense", score: float = 1.0) -> RetrievedDoc:
    chunk = Chunk(
        chunk_id=chunk_id,
        parent_id=None,
        document_id=f"doc_{chunk_id}",
        document_type="Case",
        content=f"Content of {chunk_id}",
        metadata={},
    )
    return RetrievedDoc(chunk=chunk, score=score, source=source)


@pytest.mark.unit
class TestRRF:
    """Hand-computed RRF score validation (k=60 per spec §7.2)."""

    def test_single_list_preserves_order(self):
        docs = [_make_doc("A"), _make_doc("B"), _make_doc("C")]
        result = reciprocal_rank_fusion([docs], k=60)
        # rank 0 → 1/(60+1)=0.01639, rank 1 → 1/(60+2)=0.01613, rank 2 → 1/(60+3)=0.01587
        assert [r.chunk.chunk_id for r in result] == ["A", "B", "C"]

    def test_rrf_scores_hand_computed(self):
        """
        Two lists:  dense=[A, B, C], sparse=[B, A, C]
        Hand-computed scores (k=60):
          A: 1/(60+1) + 1/(60+2) = 0.016393 + 0.016129 = 0.032522
          B: 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522   (same total, rank tie)
          C: 1/(60+3) + 1/(60+3) = 0.015873 + 0.015873 = 0.031746
        So A and B tie; C is last.
        """
        dense = [_make_doc("A", "dense"), _make_doc("B", "dense"), _make_doc("C", "dense")]
        sparse = [_make_doc("B", "sparse"), _make_doc("A", "sparse"), _make_doc("C", "sparse")]

        result = reciprocal_rank_fusion([dense, sparse], k=60)
        scores = {r.chunk.chunk_id: r.score for r in result}

        expected_A = 1 / (60 + 1) + 1 / (60 + 2)
        expected_B = 1 / (60 + 2) + 1 / (60 + 1)
        expected_C = 1 / (60 + 3) + 1 / (60 + 3)

        assert abs(scores["A"] - expected_A) < 1e-9
        assert abs(scores["B"] - expected_B) < 1e-9
        assert abs(scores["C"] - expected_C) < 1e-9
        # C must be last
        assert result[-1].chunk.chunk_id == "C"

    def test_deduplication(self):
        """Same chunk_id appearing in both lists produces only one result."""
        dense = [_make_doc("X", "dense"), _make_doc("Y", "dense")]
        sparse = [_make_doc("X", "sparse"), _make_doc("Z", "sparse")]

        result = reciprocal_rank_fusion([dense, sparse], k=60)
        ids = [r.chunk.chunk_id for r in result]
        assert ids.count("X") == 1

    def test_empty_lists_return_empty(self):
        assert reciprocal_rank_fusion([], k=60) == []
        assert reciprocal_rank_fusion([[]], k=60) == []

    def test_k_changes_relative_scores(self):
        """Larger k → smaller differences between ranks."""
        docs_k1 = [_make_doc("A"), _make_doc("B")]
        docs_k2 = [_make_doc("A"), _make_doc("B")]

        r_small = reciprocal_rank_fusion([docs_k1], k=1)
        r_large = reciprocal_rank_fusion([docs_k2], k=1000)

        diff_small = r_small[0].score - r_small[1].score
        diff_large = r_large[0].score - r_large[1].score
        assert diff_small > diff_large, "Larger k should compress rank differences"

    def test_doc_appearing_only_in_one_list(self):
        """A doc in only one list should still appear in results."""
        dense = [_make_doc("A", "dense"), _make_doc("B", "dense")]
        sparse = [_make_doc("C", "sparse")]

        result = reciprocal_rank_fusion([dense, sparse], k=60)
        ids = {r.chunk.chunk_id for r in result}
        assert "A" in ids and "B" in ids and "C" in ids

    def test_output_sorted_descending(self):
        """Result must always be sorted by descending RRF score."""
        import random
        random.seed(42)
        docs = [_make_doc(str(i)) for i in range(10)]
        random.shuffle(docs)
        result = reciprocal_rank_fusion([docs], k=60)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)
