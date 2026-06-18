"""
Startup configuration with fail-fast validation.
All secrets come from .env — never hardcoded here.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Required secrets ---
    openai_api_key: str = Field(..., min_length=1)
    google_api_key: str = Field(..., min_length=1)
    cohere_api_key: str = Field(..., min_length=1)
    tavily_api_key: str = Field(..., min_length=1)
    neo4j_password: str = Field(..., min_length=1)

    # --- Infrastructure ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    qdrant_url: str = "http://localhost:6333"

    # --- Embeddings ---
    qdrant_collection: str = "lexgraph_chunks"
    embedding_model: str = "text-embedding-3-large"
    # Native dimension for text-embedding-3-large is 3072.
    # Use 1536 only for intentional dimensionality reduction (see spec §0.4).
    embedding_dimensions: int = 3072

    # --- Generation ---
    generation_model: str = "gemini-1.5-pro"
    generation_fallback_model: str = "gpt-4o"
    generation_temperature: float = 0.1
    generation_max_tokens: int = 4096

    # --- Retrieval ---
    retrieval_top_k: int = 15
    rrf_k: int = 60

    # --- Agentic loop ---
    max_iterations: int = 3
    crag_confidence_threshold: float = 0.7

    # --- Chunking ---
    parent_chunk_size: int = 1200
    child_chunk_size: int = 150
    chunk_overlap: int = 25

    @model_validator(mode="after")
    def _validate_chunk_sizes(self) -> "Settings":
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
    """Return the singleton settings instance. Raises on first call if any required key is missing."""
    return Settings()
