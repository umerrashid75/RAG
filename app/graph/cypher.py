"""
Parametrized Cypher query builders (§6.1).
NEVER use f-string interpolation for Cypher — use parameters to prevent injection.
"""
from __future__ import annotations


def find_patent_with_claims(patent_number: str) -> tuple[str, dict]:
    """Return (cypher, params) to fetch a patent and its claims."""
    cypher = (
        "MATCH (p:Patent {patent_number: $patent_number})-[:CONTAINS]->(c:Claim) "
        "RETURN p, collect(c) AS claims"
    )
    return cypher, {"patent_number": patent_number}


def find_citing_cases(patent_number: str, hops: int = 1) -> tuple[str, dict]:
    """Return (cypher, params) for cases citing a patent (1 or 2 hops)."""
    if hops == 1:
        cypher = (
            "MATCH (case:Case)-[:ADJUDICATES]->(p:Patent {patent_number: $patent_number}) "
            "RETURN case"
        )
    else:
        cypher = (
            "MATCH (case:Case)-[:ADJUDICATES]->(p:Patent {patent_number: $patent_number}) "
            "OPTIONAL MATCH (case2:Case)-[:CITES]->(case) "
            "RETURN case, collect(case2) AS citing_cases"
        )
    return cypher, {"patent_number": patent_number}


def find_cases_by_judge(judge_name: str) -> tuple[str, dict]:
    """Return (cypher, params) for cases decided by a named judge."""
    cypher = (
        "MATCH (case:Case)-[:DECIDED_BY]->(j:Judge {name: $judge_name}) "
        "RETURN case ORDER BY case.date DESC"
    )
    return cypher, {"judge_name": judge_name}


def find_statute_cases(section_number: str) -> tuple[str, dict]:
    """Return (cypher, params) for cases interpreting a statute section."""
    cypher = (
        "MATCH (case:Case)-[:INTERPRETS]->(s:Statute {section_number: $section_number}) "
        "RETURN case ORDER BY case.date DESC"
    )
    return cypher, {"section_number": section_number}
