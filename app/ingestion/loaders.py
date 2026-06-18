"""
Data source loaders for CourtListener / USPTO bulk data (§0.3).
P0/P1: file-based loading only; API-based streaming is a future concern.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from app.ingestion.parser import DocumentType, parse_document
from app.models import Chunk

_EXTENSION_TO_TYPE: dict[str, DocumentType] = {
    ".pdf": "Case",
    ".xml": "Patent",
}


def load_directory(
    directory: Path,
    *,
    document_type: DocumentType | None = None,
    parent_size: int = 1200,
    child_size: int = 150,
    overlap: int = 25,
) -> Iterator[Chunk]:
    """
    Walk *directory* and yield Chunks for every supported file.
    If *document_type* is None, it is inferred from the file extension
    (.pdf → Case, .xml → Patent).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        inferred_type = _EXTENSION_TO_TYPE.get(path.suffix.lower())
        if inferred_type is None:
            continue
        dtype = document_type or inferred_type
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
