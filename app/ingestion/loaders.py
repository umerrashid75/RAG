"""
File-based data source loaders for local research-paper corpora.
For fetching directly from arXiv, see app/ingestion/arxiv_loader.py.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from app.ingestion.parser import parse_document
from app.models import Chunk, DocumentType

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_directory(
    directory: Path,
    *,
    document_type: DocumentType | None = None,
    parent_size: int = 1200,
    child_size: int = 150,
    overlap: int = 25,
) -> Iterator[Chunk]:
    """
    Walk *directory* and yield Chunks for every supported file (.pdf, .txt, .md).
    Files are treated as 'Paper' unless *document_type* overrides it.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        dtype = document_type or "Paper"
        try:
            chunks = parse_document(
                path,
                dtype,
                parent_size=parent_size,
                child_size=child_size,
                overlap=overlap,
            )
            yield from chunks
        except (RuntimeError, ValueError) as exc:
            # Log and continue so one bad file doesn't halt the whole ingest.
            import logging
            logging.getLogger(__name__).warning("Skipping %s: %s", path, exc)
