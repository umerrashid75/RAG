"""
Hierarchical document parser (§5.1).
Produces parent/child Chunk objects from PDF (case law) and XML (patents).
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from app.ingestion.chunker import build_chunks
from app.models import Chunk

DocumentType = Literal["Case", "Patent", "Statute"]


def _extract_pdf_text(path: Path) -> str:
    try:
        import pypdf  # type: ignore[import-untyped]

        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF {path}: {exc}") from exc


def _extract_patent_xml_text(path: Path) -> tuple[str, dict]:
    """Parse USPTO XML and extract abstract + claims as structured text."""
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse patent XML {path}: {exc}") from exc

    def _find_text(tag: str) -> str:
        el = root.find(f".//{tag}")
        return "".join(el.itertext()).strip() if el is not None else ""

    patent_number = _find_text("publication-reference") or path.stem
    abstract = _find_text("abstract")
    description = _find_text("description")
    claims_raw = [
        "".join(el.itertext()).strip()
        for el in root.findall(".//claim")
    ]
    claims_text = "\n".join(f"Claim {i+1}: {c}" for i, c in enumerate(claims_raw))

    full_text = "\n\n".join(filter(None, [abstract, description, claims_text]))
    metadata = {
        "patent_number": patent_number,
        "num_claims": len(claims_raw),
    }
    return full_text, metadata


def parse_document(
    path: Path,
    document_type: DocumentType,
    *,
    document_id: str | None = None,
    extra_metadata: dict | None = None,
    parent_size: int = 1200,
    child_size: int = 150,
    overlap: int = 25,
) -> list[Chunk]:
    """
    Parse *path* into hierarchical Chunks.

    :param path: File to parse (.pdf for cases/statutes, .xml for patents).
    :param document_type: One of 'Case', 'Patent', 'Statute'.
    :param document_id: Stable ID for the document (e.g., citation string or patent number).
    :param extra_metadata: Extra metadata merged into every chunk's metadata dict.
    :param parent_size: Token size for parent chunks.
    :param child_size: Token size for child chunks.
    :param overlap: Token overlap between consecutive chunks.
    :returns: Flat list of Chunk objects (parents then children).
    :raises RuntimeError: If the file cannot be parsed.
    :raises ValueError: If the path does not exist.
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
    elif suffix == ".xml":
        text, xml_meta = _extract_patent_xml_text(path)
        base_metadata.update(xml_meta)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .pdf or .xml")

    return build_chunks(
        text=text,
        document_id=doc_id,
        document_type=document_type,
        metadata=base_metadata,
        parent_size=parent_size,
        child_size=child_size,
        overlap=overlap,
    )
