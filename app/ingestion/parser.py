"""
Hierarchical document parser for research papers.
Produces parent/child Chunk objects from PDF (e.g. arXiv) and plain-text sources.
"""
from __future__ import annotations

from pathlib import Path

from app.ingestion.chunker import build_chunks
from app.models import Chunk, DocumentType


def _extract_pdf_text(path: Path) -> str:
    try:
        import pypdf  # type: ignore[import-untyped]

        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF {path}: {exc}") from exc


def parse_document(
    path: Path,
    document_type: DocumentType = "Paper",
    *,
    document_id: str | None = None,
    extra_metadata: dict | None = None,
    parent_size: int = 1200,
    child_size: int = 150,
    overlap: int = 25,
) -> list[Chunk]:
    """
    Parse *path* into hierarchical Chunks.

    :param path: File to parse (.pdf for papers, .txt/.md for plain text).
    :param document_type: One of 'Paper', 'Survey', 'Benchmark'.
    :param document_id: Stable ID for the document (e.g. an arXiv id).
    :param extra_metadata: Extra metadata merged into every chunk's metadata dict.
    :param parent_size: Token size for parent chunks.
    :param child_size: Token size for child chunks.
    :param overlap: Token overlap between consecutive chunks.
    :returns: Flat list of Chunk objects (parents then children).
    :raises RuntimeError: If the file cannot be parsed.
    :raises ValueError: If the path does not exist or the format is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Document path does not exist: {path}")

    doc_id = document_id or path.stem
    base_metadata: dict = {"source_path": str(path), "document_type": document_type}
    if extra_metadata:
        base_metadata.update(extra_metadata)

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf_text(path)
    elif suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .pdf, .txt, or .md")

    return build_chunks(
        text=text,
        document_id=doc_id,
        document_type=document_type,
        metadata=base_metadata,
        parent_size=parent_size,
        child_size=child_size,
        overlap=overlap,
    )
