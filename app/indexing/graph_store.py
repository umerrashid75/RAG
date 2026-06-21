"""
Neo4j graph store — schema, entity/edge upsert, health check (§6.1).
Phase P2. Constraints are created idempotently on first connect.
"""
from __future__ import annotations

import logging

from app.models import Entity, ExtractionResult, Relation

log = logging.getLogger(__name__)

# Constraints to be created on startup (idempotent IF NOT EXISTS).
_CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Benchmark) REQUIRE b.name IS UNIQUE",
]


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("neo4j package is required for Neo4jGraphStore") from exc

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def ensure_schema(self) -> None:
        with self._driver.session() as session:
            for constraint in _CONSTRAINTS:
                session.run(constraint)
        log.info("Neo4j schema constraints ensured.")

    def upsert_extraction(self, result: ExtractionResult) -> None:
        """Write entities and relations from one extraction result into the graph."""
        with self._driver.session() as session:
            for entity in result.entities:
                self._upsert_entity(session, entity)
            for relation in result.relations:
                self._upsert_relation(session, relation)

    def _upsert_entity(self, session, entity: Entity) -> None:
        # Parametrized Cypher — never string interpolation.
        cypher = (
            f"MERGE (n:{entity.type} {{name: $name}}) "
            "SET n += $attrs"
        )
        session.run(cypher, name=entity.name, attrs=entity.attributes)

    def _upsert_relation(self, session, relation: Relation) -> None:
        cypher = (
            "MATCH (src {name: $src}), (dst {name: $dst}) "
            f"MERGE (src)-[r:{relation.relation}]->(dst) "
            "SET r += $attrs"
        )
        session.run(
            cypher,
            src=relation.src,
            dst=relation.dst,
            attrs=relation.attributes,
        )

    def health(self) -> bool:
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._driver.close()
