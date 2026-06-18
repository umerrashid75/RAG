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
    title="LexGraph-RAG",
    description="Advanced Legal Case Law & Patent Intelligence System",
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
        from app.indexing.vector_store import QdrantHybridStore
        cfg = _get_settings()
        store = QdrantHybridStore(
            url=cfg.qdrant_url,
            collection=cfg.qdrant_collection,
            dimensions=cfg.embedding_dimensions,
        )
        _stores["vector_store"] = store
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
    path: str = Field(description="Absolute path to a file or directory to ingest.")
    document_type: str | None = Field(
        default=None,
        description="Override document type: Case, Patent, or Statute.",
    )


class IngestResponse(BaseModel):
    ingested_chunks: int
    document_ids: list[str]


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
def ingest(req: IngestRequest) -> IngestResponse:
    """Parse documents and upsert chunks into Qdrant."""
    from app.indexing.embeddings import OpenAIEmbeddings
    from app.ingestion.loaders import load_directory
    from app.ingestion.parser import parse_document

    cfg = _get_settings()
    store = _get_vector_store()
    store.ensure_collection()

    embeddings = OpenAIEmbeddings(
        model=cfg.embedding_model,
        dimensions=cfg.embedding_dimensions,
    )

    target = Path(req.path)
    chunks = []
    if target.is_dir():
        chunks = list(load_directory(target, document_type=req.document_type))
    elif target.is_file():
        dtype = req.document_type or ("Patent" if target.suffix == ".xml" else "Case")
        chunks = parse_document(target, dtype)
    else:
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    if not chunks:
        return IngestResponse(ingested_chunks=0, document_ids=[])

    texts = [c.content for c in chunks]
    vectors = embeddings.embed(texts)
    store.upsert(chunks, vectors)

    doc_ids = list({c.document_id for c in chunks})
    return IngestResponse(ingested_chunks=len(chunks), document_ids=doc_ids)


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="Legal question to answer.")
    top_k: int = Field(default=15, ge=1, le=100)


class Citation(BaseModel):
    document_id: str
    document_type: str
    score: float
    source: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    crag_confidence: float | None = None
    iterations: int | None = None


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(req: QueryRequest) -> QueryResponse:
    """Answer a legal query using hybrid retrieval and generation."""
    from app.indexing.embeddings import OpenAIEmbeddings
    from app.indexing.vector_store import QdrantHybridStore
    from app.llm.providers import GeminiProvider
    from app.retrieval.hybrid import HybridRetriever

    cfg = _get_settings()
    store = _get_vector_store()

    embeddings = OpenAIEmbeddings(
        model=cfg.embedding_model,
        dimensions=cfg.embedding_dimensions,
    )
    retriever = HybridRetriever(store, embeddings, rrf_k=cfg.rrf_k)
    llm = GeminiProvider(
        api_key=cfg.google_api_key,
        model=cfg.generation_model,
        temperature=cfg.generation_temperature,
        max_tokens=cfg.generation_max_tokens,
    )

    docs = retriever.retrieve(req.query, top_k=req.top_k)

    context = "\n---\n".join(
        f"[{d.chunk.document_type} | {d.chunk.document_id}]\n{d.chunk.content}"
        for d in docs
    )
    prompt = (
        "You are a legal intelligence assistant. Answer the following query using ONLY "
        "the provided context. Cite every factual claim with the document_id.\n\n"
        f"Query: {req.query}\n\nContext:\n{context}\n\nAnswer (with citations):"
    )
    answer = str(llm.generate(prompt))

    citations = [
        Citation(
            document_id=d.chunk.document_id,
            document_type=d.chunk.document_type,
            score=round(d.score, 4),
            source=d.source,
        )
        for d in docs[:10]
    ]

    return QueryResponse(answer=answer, citations=citations)
