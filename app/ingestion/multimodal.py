"""
Multimodal diagram captioning via Vision LLM (§5.2).
Phase P5 — deferred. Stub only.
"""
from __future__ import annotations

from pathlib import Path

from app.models import Chunk


def caption_diagram(image_path: Path, document_id: str) -> Chunk:
    """P5 stub. Will use Gemini Vision to caption diagrams in legal exhibits."""
    raise NotImplementedError("Multimodal captioning is deferred to Phase P5.")
