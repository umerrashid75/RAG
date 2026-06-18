"""
Typer CLI entry point: `lexgraph ingest <path>`, `lexgraph index --rebuild`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="lexgraph",
    help="LexGraph-RAG: Legal Case Law & Patent Intelligence CLI",
    add_completion=False,
)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest."),
    document_type: Optional[str] = typer.Option(
        None, "--type", "-t",
        help="Override document type: Case, Patent, or Statute.",
    ),
) -> None:
    """Parse and index documents into Qdrant."""
    from app.config import get_settings
    from app.indexing.embeddings import OpenAIEmbeddings
    from app.indexing.vector_store import QdrantHybridStore
    from app.ingestion.loaders import load_directory
    from app.ingestion.parser import parse_document

    cfg = get_settings()
    store = QdrantHybridStore(
        url=cfg.qdrant_url,
        collection=cfg.qdrant_collection,
        dimensions=cfg.embedding_dimensions,
    )
    store.ensure_collection()

    embeddings = OpenAIEmbeddings(
        model=cfg.embedding_model,
        dimensions=cfg.embedding_dimensions,
    )

    if path.is_dir():
        chunks = list(load_directory(path, document_type=document_type))
    elif path.is_file():
        dtype = document_type or ("Patent" if path.suffix == ".xml" else "Case")
        chunks = parse_document(path, dtype)
    else:
        typer.echo(f"Error: path not found: {path}", err=True)
        raise typer.Exit(code=1)

    if not chunks:
        typer.echo("No chunks produced — check the file format.")
        return

    typer.echo(f"Ingesting {len(chunks)} chunks from {path}...")
    texts = [c.content for c in chunks]
    vectors = embeddings.embed(texts)
    store.upsert(chunks, vectors)

    doc_ids = {c.document_id for c in chunks}
    typer.echo(f"Done. Indexed {len(chunks)} chunks across {len(doc_ids)} document(s).")


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="Drop and recreate the Qdrant collection."),
) -> None:
    """Manage the Qdrant vector index."""
    from app.config import get_settings
    from app.indexing.vector_store import QdrantHybridStore

    cfg = get_settings()
    store = QdrantHybridStore(
        url=cfg.qdrant_url,
        collection=cfg.qdrant_collection,
        dimensions=cfg.embedding_dimensions,
    )

    if rebuild:
        try:
            store._client.delete_collection(cfg.qdrant_collection)
            typer.echo(f"Dropped collection '{cfg.qdrant_collection}'.")
        except Exception:
            pass

    store.ensure_collection()
    typer.echo(f"Collection '{cfg.qdrant_collection}' is ready.")


@app.command()
def query(
    question: str = typer.Argument(..., help="Legal question to answer."),
    top_k: int = typer.Option(15, "--top-k", help="Number of chunks to retrieve."),
) -> None:
    """Query the system and print the answer with citations."""
    from app.config import get_settings
    from app.indexing.embeddings import OpenAIEmbeddings
    from app.indexing.vector_store import QdrantHybridStore
    from app.llm.providers import GeminiProvider
    from app.retrieval.hybrid import HybridRetriever

    cfg = get_settings()
    store = QdrantHybridStore(
        url=cfg.qdrant_url,
        collection=cfg.qdrant_collection,
        dimensions=cfg.embedding_dimensions,
    )
    embeddings = OpenAIEmbeddings(cfg.embedding_model, cfg.embedding_dimensions)
    retriever = HybridRetriever(store, embeddings, rrf_k=cfg.rrf_k)
    llm = GeminiProvider(cfg.google_api_key, cfg.generation_model)

    docs = retriever.retrieve(question, top_k=top_k)
    context = "\n---\n".join(
        f"[{d.chunk.document_type} | {d.chunk.document_id}]\n{d.chunk.content}"
        for d in docs
    )
    prompt = (
        "Answer the following legal query using ONLY the context below. "
        "Cite each claim by document_id.\n\n"
        f"Query: {question}\n\nContext:\n{context}\n\nAnswer:"
    )
    from app.llm.providers import GeminiProvider as _G
    answer = str(llm.generate(prompt))

    typer.echo("\n=== Answer ===")
    typer.echo(answer)
    typer.echo("\n=== Citations ===")
    for d in docs[:10]:
        typer.echo(f"  [{d.source}] {d.chunk.document_type}/{d.chunk.document_id}  score={d.score:.4f}")


if __name__ == "__main__":
    app()
