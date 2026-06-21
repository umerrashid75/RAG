"""
Startup configuration with fail-fast validation.
All secrets come from .env — never hardcoded here.

The default profile is free-tier: local FastEmbed embeddings (no API key) plus
Groq for generation. OpenAI / Gemini / Cohere / Tavily are optional upgrades —
the app boots and serves vector RAG even when only GROQ_API_KEY is set.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Provider selection ---
    llm_provider: Literal["groq", "gemini", "openai"] = "groq"
    embedding_provider: Literal["fastembed", "openai"] = "fastembed"

    # --- API keys (all optional; only the active provider's key is required) ---
    groq_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    cohere_api_key: str = ""
    tavily_api_key: str = ""

    # --- Infrastructure ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    # Graph enrichment is optional; the app degrades to vector RAG without it.
    neo4j_password: str = "papergraph_dev"  # noqa: S105 — local dev default, override in prod
    qdrant_url: str = "http://localhost:6333"
    # Required only for managed Qdrant Cloud; empty for local/self-hosted.
    qdrant_api_key: str = ""

    # --- Embeddings ---
    qdrant_collection: str = "papergraph_chunks"
    # Default: local FastEmbed model (ONNX, no API key, 384-dim).
    # For OpenAI set embedding_provider=openai, model=text-embedding-3-large, dims=3072.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    # --- Generation ---
    # Groq production model with JSON-mode support. If Groq deprecates it,
    # update via GENERATION_MODEL env var (see https://console.groq.com/docs/models).
    generation_model: str = "llama-3.3-70b-versatile"
    generation_temperature: float = 0.1
    generation_max_tokens: int = 4096

    # --- Retrieval ---
    retrieval_top_k: int = 15
    rrf_k: int = 60

    # --- Graph enrichment ---
    # When true, ingestion runs LLM entity/relation extraction into Neo4j.
    # Off by default so free-tier ingestion stays fast and key-free.
    enable_graph: bool = False

    # --- Agentic loop ---
    max_iterations: int = 3
    crag_confidence_threshold: float = 0.7

    # --- Chunking ---
    parent_chunk_size: int = 1200
    child_chunk_size: int = 150
    chunk_overlap: int = 25

    def active_llm_key(self) -> str:
        """Return the API key for the configured LLM provider, or raise a clear error."""
        key_map = {
            "groq": ("groq_api_key", self.groq_api_key),
            "gemini": ("google_api_key", self.google_api_key),
            "openai": ("openai_api_key", self.openai_api_key),
        }
        env_name, value = key_map[self.llm_provider]
        if not value:
            raise ValueError(
                f"LLM_PROVIDER='{self.llm_provider}' requires {env_name.upper()} to be set. "
                f"Get a free Groq key at https://console.groq.com and add it to your .env."
            )
        return value

    @model_validator(mode="after")
    def _validate_chunk_sizes(self) -> Settings:
        if self.child_chunk_size >= self.parent_chunk_size:
            raise ValueError(
                f"child_chunk_size ({self.child_chunk_size}) must be "
                f"< parent_chunk_size ({self.parent_chunk_size})"
            )
        if self.chunk_overlap >= self.child_chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"< child_chunk_size ({self.child_chunk_size})"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance (validated on first access)."""
    return Settings()
