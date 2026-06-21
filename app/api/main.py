"""
FastAPI application: /healthz, /query, /ingest endpoints.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

app = FastAPI(
    title="PaperGraph-RAG",
    description="Agentic GraphRAG for understanding the latest AI research papers",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Lazy singletons — initialized on first request to avoid startup errors
# when keys/services are not yet available.
# ---------------------------------------------------------------------------

_stores: dict[str, Any] = {}


def _get_settings():
    from app.config import get_settings
    return get_settings()


def _get_vector_store():
    if "vector_store" not in _stores:
        from app.factory import build_vector_store
        _stores["vector_store"] = build_vector_store(_get_settings())
    return _stores["vector_store"]


def _get_graph_store():
    if "graph_store" not in _stores:
        from app.indexing.graph_store import Neo4jGraphStore
        cfg = _get_settings()
        store = Neo4jGraphStore(
            uri=cfg.neo4j_uri,
            user=cfg.neo4j_user,
            password=cfg.neo4j_password,
        )
        _stores["graph_store"] = store
    return _stores["graph_store"]


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["ops"])
def healthz() -> dict[str, Any]:
    """Return connectivity status for Qdrant and Neo4j."""
    vector_ok = False
    graph_ok = False

    try:
        vector_ok = _get_vector_store().health()
    except Exception as exc:
        log.warning("Qdrant health check failed: %s", exc)

    try:
        graph_ok = _get_graph_store().health()
    except Exception as exc:
        log.warning("Neo4j health check failed: %s", exc)

    status = "ok" if (vector_ok and graph_ok) else "degraded"
    return {
        "status": status,
        "stores": {
            "qdrant": "ok" if vector_ok else "unreachable",
            "neo4j": "ok" if graph_ok else "unreachable",
        },
    }


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    path: str | None = Field(
        default=None,
        description="Path to a local file or directory of papers (.pdf/.txt/.md).",
    )
    arxiv: str | None = Field(
        default=None,
        description="arXiv id(s) or a search query to fetch and ingest directly.",
    )
    max_results: int = Field(default=5, ge=1, le=50, description="Max papers for an arXiv search.")
    document_type: str | None = Field(
        default=None,
        description="Override document type: Paper, Survey, or Benchmark.",
    )


class IngestResponse(BaseModel):
    ingested_chunks: int
    document_ids: list[str]


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
def ingest(req: IngestRequest) -> IngestResponse:
    """Fetch/parse research papers and upsert their chunks into Qdrant."""
    from app.factory import build_embeddings
    from app.ingestion.arxiv_loader import fetch_arxiv
    from app.ingestion.loaders import load_directory
    from app.ingestion.parser import parse_document

    cfg = _get_settings()
    store = _get_vector_store()
    store.ensure_collection()

    embeddings = build_embeddings(cfg)

    chunks = []
    if req.arxiv:
        chunks = list(fetch_arxiv(req.arxiv, max_results=req.max_results))
    elif req.path:
        target = Path(req.path)
        if target.is_dir():
            chunks = list(load_directory(target, document_type=req.document_type))
        elif target.is_file():
            chunks = parse_document(target, req.document_type or "Paper")
        else:
            raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
    else:
        raise HTTPException(status_code=400, detail="Provide either 'arxiv' or 'path'.")

    if not chunks:
        return IngestResponse(ingested_chunks=0, document_ids=[])

    texts = [c.content for c in chunks]
    vectors = embeddings.embed(texts)
    store.upsert(chunks, vectors)

    if cfg.enable_graph:
        from app.factory import build_graph_store, build_llm
        from app.graph.enrich import enrich_graph

        try:
            enrich_graph(chunks, build_llm(cfg), build_graph_store(cfg))
        except Exception:
            log.exception("Graph enrichment failed; vector index is still populated.")

    doc_ids = list({c.document_id for c in chunks})
    return IngestResponse(ingested_chunks=len(chunks), document_ids=doc_ids)


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="Research question to answer.")
    top_k: int = Field(default=15, ge=1, le=100)


class Citation(BaseModel):
    document_id: str
    document_type: str
    score: float
    source: str
    title: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    crag_confidence: float | None = None
    iterations: int | None = None
    reflections: dict[str, Any] | None = None


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(req: QueryRequest) -> QueryResponse:
    """
    Answer a research question via the agentic loop: route → hybrid retrieve →
    CRAG grade → generate → Self-RAG reflect → (loop or finalize).
    """
    from app.factory import build_agentic_graph, build_retriever

    cfg = _get_settings()
    retriever = build_retriever(cfg, store=_get_vector_store())
    graph = build_agentic_graph(cfg, retriever=retriever)

    initial_state: dict[str, Any] = {
        "query": req.query,
        "retrieved": [],
        "crag_confidence": 0.0,
        "draft": "",
        "reflections": {},
        "iterations": 0,
        "answer": None,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:  # surface provider/config errors as a clean 502
        log.exception("Agentic query failed")
        raise HTTPException(status_code=502, detail=f"Query failed: {exc}") from exc

    docs = final_state.get("retrieved", [])
    citations = [
        Citation(
            document_id=d.chunk.document_id,
            document_type=d.chunk.document_type,
            score=round(d.score, 4),
            source=d.source,
            title=d.chunk.metadata.get("title"),
        )
        for d in docs[:10]
    ]

    return QueryResponse(
        answer=final_state.get("answer") or final_state.get("draft") or "",
        citations=citations,
        crag_confidence=final_state.get("crag_confidence"),
        iterations=final_state.get("iterations"),
        reflections=final_state.get("reflections") or None,
    )
