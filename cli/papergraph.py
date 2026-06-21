"""
Typer CLI entry point for PaperGraph-RAG.

Commands:
    papergraph arxiv "<query or ids>"   fetch papers from arXiv and index them
    papergraph ingest <path>            index local PDFs / text files
    papergraph index --rebuild          (re)create the Qdrant collection
    papergraph query "<question>"       run the agentic RAG loop and print the answer
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="papergraph",
    help="PaperGraph-RAG: agentic GraphRAG for the latest AI research papers.",
    add_completion=False,
)


def _index_chunks(chunks: list) -> None:
    """Embed and upsert a list of chunks into Qdrant. Shared by arxiv/ingest."""
    from app.config import get_settings
    from app.factory import build_embeddings, build_vector_store

    if not chunks:
        typer.echo("No chunks produced — check the source.")
        return

    cfg = get_settings()
    store = build_vector_store(cfg)
    store.ensure_collection()
    embeddings = build_embeddings(cfg)

    typer.echo(f"Embedding and indexing {len(chunks)} chunks...")
    vectors = embeddings.embed([c.content for c in chunks])
    store.upsert(chunks, vectors)

    doc_ids = {c.document_id for c in chunks}
    typer.echo(f"Done. Indexed {len(chunks)} chunks across {len(doc_ids)} paper(s).")

    if cfg.enable_graph:
        from app.factory import build_graph_store, build_llm
        from app.graph.enrich import enrich_graph

        typer.echo("Enriching knowledge graph (entity/relation extraction)...")
        enriched = enrich_graph(chunks, build_llm(cfg), build_graph_store(cfg))
        typer.echo(f"Graph-enriched {enriched} paper(s).")


@app.command()
def arxiv(
    query: str = typer.Argument(..., help="arXiv id(s) or a search query."),
    max_results: int = typer.Option(5, "--max", "-n", help="Max papers for a search query."),
) -> None:
    """Fetch papers from arXiv and index them (no paid API key required)."""
    from app.ingestion.arxiv_loader import fetch_arxiv

    typer.echo(f"Fetching from arXiv: {query!r}")
    _index_chunks(list(fetch_arxiv(query, max_results=max_results)))


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest (.pdf/.txt/.md)."),
    document_type: str | None = typer.Option(
        None, "--type", "-t", help="Override document type: Paper, Survey, or Benchmark."
    ),
) -> None:
    """Parse and index local research papers into Qdrant."""
    from app.ingestion.loaders import load_directory
    from app.ingestion.parser import parse_document

    if path.is_dir():
        chunks = list(load_directory(path, document_type=document_type))
    elif path.is_file():
        chunks = parse_document(path, document_type or "Paper")
    else:
        typer.echo(f"Error: path not found: {path}", err=True)
        raise typer.Exit(code=1)

    _index_chunks(chunks)


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="Drop and recreate the collection."),
) -> None:
    """Manage the Qdrant vector index."""
    from app.config import get_settings
    from app.factory import build_vector_store

    cfg = get_settings()
    store = build_vector_store(cfg)

    if rebuild:
        try:
            store._client.delete_collection(cfg.qdrant_collection)
            typer.echo(f"Dropped collection '{cfg.qdrant_collection}'.")
        except Exception as exc:  # noqa: BLE001 — drop is best-effort
            typer.echo(f"(No existing collection to drop: {exc})")

    store.ensure_collection()
    typer.echo(f"Collection '{cfg.qdrant_collection}' is ready.")


@app.command()
def query(
    question: str = typer.Argument(..., help="Research question to answer."),
    top_k: int = typer.Option(15, "--top-k", help="Number of chunks to retrieve."),
) -> None:
    """Run the agentic CRAG + Self-RAG loop and print the answer with citations."""
    from app.config import get_settings
    from app.factory import build_agentic_graph, build_retriever

    cfg = get_settings()
    retriever = build_retriever(cfg)
    graph = build_agentic_graph(cfg, retriever=retriever)

    final_state = graph.invoke(
        {
            "query": question,
            "retrieved": [],
            "crag_confidence": 0.0,
            "draft": "",
            "reflections": {},
            "iterations": 0,
            "answer": None,
        }
    )

    typer.echo("\n=== Answer ===")
    typer.echo(final_state.get("answer") or final_state.get("draft") or "(no answer)")

    typer.echo("\n=== Citations ===")
    for d in final_state.get("retrieved", [])[:10]:
        title = d.chunk.metadata.get("title", "")
        typer.echo(f"  [{d.source}] {d.chunk.document_id}  score={d.score:.4f}  {title}")

    conf = final_state.get("crag_confidence")
    iters = final_state.get("iterations")
    typer.echo(f"\nCRAG confidence: {conf}  |  iterations: {iters}")


if __name__ == "__main__":
    app()
