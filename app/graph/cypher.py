"""
Parametrized Cypher query builders for the research-paper knowledge graph.
NEVER use f-string interpolation for values — use parameters to prevent injection.
"""
from __future__ import annotations


def find_paper_with_methods(paper_name: str) -> tuple[str, dict]:
    """Return (cypher, params) to fetch a paper and the methods it proposes."""
    cypher = (
        "MATCH (p:Paper {name: $paper_name})-[:PROPOSES]->(m:Method) "
        "RETURN p, collect(m) AS methods"
    )
    return cypher, {"paper_name": paper_name}


def find_citing_papers(paper_name: str, hops: int = 1) -> tuple[str, dict]:
    """Return (cypher, params) for papers that cite a given paper (1 or 2 hops)."""
    if hops == 1:
        cypher = (
            "MATCH (citing:Paper)-[:CITES]->(p:Paper {name: $paper_name}) "
            "RETURN citing"
        )
    else:
        cypher = (
            "MATCH (citing:Paper)-[:CITES]->(p:Paper {name: $paper_name}) "
            "OPTIONAL MATCH (downstream:Paper)-[:CITES]->(citing) "
            "RETURN citing, collect(downstream) AS downstream_papers"
        )
    return cypher, {"paper_name": paper_name}


def find_papers_by_author(author_name: str) -> tuple[str, dict]:
    """Return (cypher, params) for papers written by a named author."""
    cypher = (
        "MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author {name: $author_name}) "
        "RETURN p ORDER BY p.published DESC"
    )
    return cypher, {"author_name": author_name}


def find_papers_using_dataset(dataset_name: str) -> tuple[str, dict]:
    """Return (cypher, params) for papers that evaluate on a given dataset/benchmark."""
    cypher = (
        "MATCH (p:Paper)-[:EVALUATES_ON]->(d {name: $dataset_name}) "
        "RETURN p ORDER BY p.published DESC"
    )
    return cypher, {"dataset_name": dataset_name}
