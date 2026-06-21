"""
Tests for parametrized Cypher builders — verifies injection-safety.
No Neo4j connection required.
"""
import pytest

from app.graph.cypher import (
    find_citing_papers,
    find_paper_with_methods,
    find_papers_by_author,
    find_papers_using_dataset,
)


@pytest.mark.unit
class TestCypherBuilders:
    def test_paper_query_is_parametrized(self):
        cypher, params = find_paper_with_methods("Attention Is All You Need")
        assert "$paper_name" in cypher
        # Value must NOT be interpolated into the query string.
        assert "Attention Is All You Need" not in cypher
        assert params["paper_name"] == "Attention Is All You Need"

    def test_citing_papers_1hop(self):
        cypher, params = find_citing_papers("2310.06825", hops=1)
        assert "$paper_name" in cypher
        assert params["paper_name"] == "2310.06825"
        assert "downstream" not in cypher

    def test_citing_papers_2hop(self):
        cypher, _ = find_citing_papers("2310.06825", hops=2)
        assert "downstream" in cypher

    def test_author_query_parametrized(self):
        cypher, params = find_papers_by_author("Ashish Vaswani")
        assert "$author_name" in cypher
        assert "Ashish Vaswani" not in cypher
        assert params["author_name"] == "Ashish Vaswani"

    def test_dataset_query_parametrized(self):
        cypher, params = find_papers_using_dataset("Natural Questions")
        assert "$dataset_name" in cypher
        assert "Natural Questions" not in cypher
        assert params["dataset_name"] == "Natural Questions"

    def test_no_fstring_injection_vector(self):
        """Attempt to inject Cypher via the paper name — must end up only in params."""
        malicious = "'; DROP DATABASE neo4j; //'"
        cypher, params = find_paper_with_methods(malicious)
        assert malicious not in cypher
        assert params["paper_name"] == malicious
