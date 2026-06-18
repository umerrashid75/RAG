"""
Tests for parametrized Cypher builders — verifies injection-safety.
No Neo4j connection required.
"""
import pytest

from app.graph.cypher import (
    find_citing_cases,
    find_cases_by_judge,
    find_patent_with_claims,
    find_statute_cases,
)


@pytest.mark.unit
class TestCypherBuilders:
    def test_patent_query_is_parametrized(self):
        cypher, params = find_patent_with_claims("US-10293847-B2")
        assert "$patent_number" in cypher
        assert "US-10293847-B2" not in cypher  # must NOT be interpolated into the query string
        assert params["patent_number"] == "US-10293847-B2"

    def test_citing_cases_1hop(self):
        cypher, params = find_citing_cases("US-99999-B2", hops=1)
        assert "$patent_number" in cypher
        assert params["patent_number"] == "US-99999-B2"
        # 1-hop should not mention case2
        assert "case2" not in cypher

    def test_citing_cases_2hop(self):
        cypher, params = find_citing_cases("US-99999-B2", hops=2)
        assert "case2" in cypher

    def test_judge_query_parametrized(self):
        cypher, params = find_cases_by_judge("Judge Smith")
        assert "$judge_name" in cypher
        assert "Judge Smith" not in cypher
        assert params["judge_name"] == "Judge Smith"

    def test_statute_query_parametrized(self):
        cypher, params = find_statute_cases("35 U.S.C. 102")
        assert "$section_number" in cypher
        assert "35 U.S.C. 102" not in cypher
        assert params["section_number"] == "35 U.S.C. 102"

    def test_no_fstring_injection_vector(self):
        """Attempt to inject Cypher via patent number — must end up only in params."""
        malicious = "'; DROP DATABASE neo4j; //'"
        cypher, params = find_patent_with_claims(malicious)
        assert malicious not in cypher
        assert params["patent_number"] == malicious
