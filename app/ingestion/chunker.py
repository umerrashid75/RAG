"""
Hierarchical parent/child chunking (§5.1).

Strategy: split text into parent chunks (1000-1500 tokens), then split each
parent into child chunks (100-200 tokens) with overlap. Retrieval searches
child embeddings for precision; generation context uses the parent.
"""
from __future__ import annotations

import hashlib
import uuid

import tiktoken

from app.models import Chunk


_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_by_tokens(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split *text* into overlapping token windows of *chunk_size* tokens.
    Returns at least one segment even if text is shorter than chunk_size.
    """
    tokens = _ENCODING.encode(text)
    if not tokens:
        return []

    segments: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        segment_tokens = tokens[start:end]
        segments.append(_ENCODING.decode(segment_tokens))
        if end == len(tokens):
            break
        start += step

    return segments


def _make_id(document_id: str, suffix: str) -> str:
    return hashlib.sha1(f"{document_id}:{suffix}".encode()).hexdigest()[:16]


def build_chunks(
    text: str,
    document_id: str,
    document_type: str,
    metadata: dict,
    *,
    parent_size: int = 1200,
    child_size: int = 150,
    overlap: int = 25,
) -> list[Chunk]:
    """
    Return a flat list of Chunk objects representing the full hierarchy.
    Parent chunks come first; each child references its parent's chunk_id.

    Invariants:
    - Every child has a non-None parent_id.
    - Every parent has parent_id=None.
    - child token count <= child_size (may be less for trailing segments).
    """
    if not text.strip():
        return []

    parent_texts = _split_by_tokens(text, parent_size, overlap)
    chunks: list[Chunk] = []

    for p_idx, p_text in enumerate(parent_texts):
        p_id = _make_id(document_id, f"p{p_idx}")
        parent = Chunk(
            chunk_id=p_id,
            parent_id=None,
            document_id=document_id,
            document_type=document_type,
            content=p_text,
            metadata={**metadata, "chunk_index": p_idx, "level": "parent"},
        )
        chunks.append(parent)

        child_texts = _split_by_tokens(p_text, child_size, overlap)
        for c_idx, c_text in enumerate(child_texts):
            c_id = _make_id(document_id, f"p{p_idx}c{c_idx}")
            child = Chunk(
                chunk_id=c_id,
                parent_id=p_id,
                document_id=document_id,
                document_type=document_type,
                content=c_text,
                metadata={**metadata, "chunk_index": c_idx, "parent_index": p_idx, "level": "child"},
            )
            chunks.append(child)

    return chunks
