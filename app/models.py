"""
Canonical data models and provider interfaces (§2 of the spec).
These are the contracts every module builds against.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


DocumentType = Literal["Paper", "Survey", "Benchmark"]


class Chunk(BaseModel):
    """A hierarchical document fragment — the atomic unit flowing through the pipeline."""

    chunk_id: str
    parent_id: str | None = None  # None for parent-level chunks
    document_id: str
    document_type: DocumentType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # arxiv_id, title, authors, published, categories, section, page, etc. live in metadata


class RetrievedDoc(BaseModel):
    """A chunk returned by any retriever, annotated with score and source."""

    chunk: Chunk
    score: float
    source: Literal["dense", "sparse", "graph", "web"]


# ---------------------------------------------------------------------------
# Provider interfaces (Protocol — swap implementations without touching callers)
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        """
        Generate text for *prompt*.
        If *schema* is given, return a parsed Pydantic instance; else return a plain string.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedDoc]:
        """Return up to *top_k* documents ranked by relevance to *query*."""
        ...


# ---------------------------------------------------------------------------
# LangGraph agent state (§2, threaded through the entire agentic loop)
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    query: str
    route: Literal["global", "local", "multi_hop"]
    retrieved: list[RetrievedDoc]
    crag_confidence: float
    draft: str
    reflections: dict[str, Any]  # Is-Rel, Is-Faith, Is-Use, Retrieve flags (§8.2)
    iterations: int              # bounded by config.max_iterations
    answer: str | None


# ---------------------------------------------------------------------------
# Graph extraction schema (used by app/graph/extractor.py, §6.2)
# ---------------------------------------------------------------------------


EntityType = Literal[
    "Paper", "Author", "Method", "Dataset", "Benchmark", "Task", "Concept", "Institution"
]


class Entity(BaseModel):
    name: str
    type: EntityType
    attributes: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    src: str
    relation: str
    dst: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
