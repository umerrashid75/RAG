"""
Graph enrichment — turn ingested chunks into a Neo4j knowledge graph.

Groups chunks by document, runs LLM entity/relation extraction on each document's
text, and upserts the result. Used by the ingest paths when ENABLE_GRAPH=true.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from app.graph.extractor import extract_entities
from app.models import Chunk, LLMProvider

log = logging.getLogger(__name__)


def enrich_graph(
    chunks: list[Chunk],
    llm: LLMProvider,
    graph_store,
    *,
    max_chars_per_doc: int = 6000,
) -> int:
    """
    Extract entities/relations per document and upsert them into the graph store.

    Returns the number of documents successfully enriched. Failures on a single
    document are logged and skipped so one bad extraction doesn't abort ingestion.
    """
    graph_store.ensure_schema()

    # Reconstruct each document's text from its parent-level chunks (ordered).
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.parent_id is None:  # parent chunks carry the readable text
            by_doc[chunk.document_id].append(chunk)

    enriched = 0
    for doc_id, parents in by_doc.items():
        text = "\n".join(p.content for p in parents)[:max_chars_per_doc]
        if not text.strip():
            continue
        try:
            result = extract_entities(text, llm)
            graph_store.upsert_extraction(result)
            enriched += 1
            log.info(
                "Graph-enriched %s: %d entities, %d relations",
                doc_id,
                len(result.entities),
                len(result.relations),
            )
        except Exception as exc:  # noqa: BLE001 — skip a bad doc, keep ingesting
            log.warning("Graph enrichment failed for %s: %s", doc_id, exc)

    return enriched
