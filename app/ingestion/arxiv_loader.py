"""
arXiv ingestion — fetch papers by id or search query, download the PDF, and parse
it into hierarchical Chunks. This is the zero-cost demo path: no paid API key needed
to populate the index with real, current AI research.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from app.ingestion.parser import parse_document
from app.models import Chunk

log = logging.getLogger(__name__)

# Survey detection is best-effort, purely from the title.
_SURVEY_HINTS = ("survey", "a review", "overview of", "systematic review")
_BENCHMARK_HINTS = ("benchmark", "evaluation suite", "leaderboard")


def _classify(title: str) -> str:
    lowered = title.lower()
    if any(h in lowered for h in _SURVEY_HINTS):
        return "Survey"
    if any(h in lowered for h in _BENCHMARK_HINTS):
        return "Benchmark"
    return "Paper"


def _results_for(query: str, max_results: int):
    """Yield arxiv.Result objects for an id list or a free-text search query."""
    try:
        import arxiv  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("arxiv is required for arXiv ingestion (pip install arxiv)") from exc

    client = arxiv.Client()
    # Heuristic: comma/space separated tokens that look like arXiv ids → id_list.
    tokens = [t.strip() for t in query.replace(",", " ").split() if t.strip()]
    looks_like_ids = all(any(c.isdigit() for c in t) and "." in t for t in tokens)

    if looks_like_ids and tokens:
        search = arxiv.Search(id_list=tokens)
    else:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
    return client.results(search)


def fetch_arxiv(
    query: str,
    *,
    max_results: int = 5,
    download_dir: Path | None = None,
    parent_size: int = 1200,
    child_size: int = 150,
    overlap: int = 25,
) -> Iterator[Chunk]:
    """
    Fetch papers from arXiv and yield parsed Chunks.

    :param query: An arXiv id, several ids, or a free-text search query.
    :param max_results: Max papers to fetch for a search query (ignored for id lists).
    :param download_dir: Where PDFs are cached (default: data/arxiv).
    """
    download_dir = Path(download_dir or "data/arxiv")
    download_dir.mkdir(parents=True, exist_ok=True)

    for result in _results_for(query, max_results):
        arxiv_id = result.get_short_id()
        title = result.title.strip().replace("\n", " ")
        pdf_path = download_dir / f"{arxiv_id.replace('/', '_')}.pdf"

        try:
            if not pdf_path.exists():
                log.info("Downloading arXiv:%s — %s", arxiv_id, title)
                result.download_pdf(dirpath=str(download_dir), filename=pdf_path.name)
        except Exception as exc:  # noqa: BLE001 — one bad download shouldn't halt the batch
            log.warning("Failed to download arXiv:%s: %s", arxiv_id, exc)
            continue

        metadata = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": [a.name for a in result.authors],
            "published": result.published.isoformat() if result.published else None,
            "categories": list(result.categories),
            "url": result.entry_id,
        }

        try:
            yield from parse_document(
                pdf_path,
                _classify(title),  # type: ignore[arg-type]
                document_id=arxiv_id,
                extra_metadata=metadata,
                parent_size=parent_size,
                child_size=child_size,
                overlap=overlap,
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to parse arXiv:%s: %s", arxiv_id, exc)
