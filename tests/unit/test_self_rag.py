"""
Tests for Self-RAG grades_pass logic (deterministic — no LLM calls).
"""
import pytest

from app.agents.self_rag import ReflectionGrades, grades_pass


@pytest.mark.unit
class TestGradesPass:
    def test_all_true_passes(self):
        grades = ReflectionGrades(
            is_relevant=True,
            is_faithful=True,
            is_useful=True,
            retrieve_again=False,
            reason="All good",
        )
        assert grades_pass(grades) is True

    def test_unfaithful_fails(self):
        grades = ReflectionGrades(
            is_relevant=True,
            is_faithful=False,
            is_useful=True,
            retrieve_again=True,
            reason="Hallucination detected",
        )
        assert grades_pass(grades) is False

    def test_irrelevant_fails(self):
        grades = ReflectionGrades(
            is_relevant=False,
            is_faithful=True,
            is_useful=True,
            retrieve_again=True,
            reason="Off-topic",
        )
        assert grades_pass(grades) is False

    def test_not_useful_fails(self):
        grades = ReflectionGrades(
            is_relevant=True,
            is_faithful=True,
            is_useful=False,
            retrieve_again=True,
            reason="Did not answer the query",
        )
        assert grades_pass(grades) is False

    def test_retrieve_again_does_not_affect_pass(self):
        """retrieve_again=True should NOT block passing — it is advisory only."""
        grades = ReflectionGrades(
            is_relevant=True,
            is_faithful=True,
            is_useful=True,
            retrieve_again=True,  # wants more retrieval but grades pass
            reason="Could improve with more docs",
        )
        assert grades_pass(grades) is True
