"""
Provider factories — turn a Settings object into the configured LLM / embedding /
retriever / agentic-graph instances. Centralizes provider selection so callers
(API, CLI, eval) stay free of branching logic.
"""
from __future__ import annotations

from app.config import Settings
from app.models import EmbeddingProvider, LLMProvider


def build_embeddings(cfg: Settings) -> EmbeddingProvider:
    """Construct the embedding provider selected in config (default: FastEmbed, free)."""
    if cfg.embedding_provider == "openai":
        from app.indexing.embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(model=cfg.embedding_model, dimensions=cfg.embedding_dimensions)

    from app.indexing.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model=cfg.embedding_model, dimensions=cfg.embedding_dimensions)


def build_llm(cfg: Settings) -> LLMProvider:
    """Construct the LLM provider selected in config (default: Groq, free-tier)."""
    api_key = cfg.active_llm_key()  # raises a clear error if the key is missing

    if cfg.llm_provider == "gemini":
        from app.llm.providers import GeminiProvider

        return GeminiProvider(
            api_key=api_key,
            model=cfg.generation_model,
            temperature=cfg.generation_temperature,
            max_tokens=cfg.generation_max_tokens,
        )

    if cfg.llm_provider == "openai":
        from app.llm.providers import OpenAIProvider

        return OpenAIProvider(
            api_key=api_key,
            model=cfg.generation_model,
            temperature=cfg.generation_temperature,
            max_tokens=cfg.generation_max_tokens,
        )

    from app.llm.providers import GroqProvider

    return GroqProvider(
        api_key=api_key,
        model=cfg.generation_model,
        temperature=cfg.generation_temperature,
        max_tokens=cfg.generation_max_tokens,
    )


def build_vector_store(cfg: Settings):
    """Construct the Qdrant hybrid store from config."""
    from app.indexing.vector_store import QdrantHybridStore

    return QdrantHybridStore(
        url=cfg.qdrant_url,
        collection=cfg.qdrant_collection,
        dimensions=cfg.embedding_dimensions,
        api_key=cfg.qdrant_api_key or None,
    )


def build_graph_store(cfg: Settings):
    """Construct the Neo4j graph store from config."""
    from app.indexing.graph_store import Neo4jGraphStore

    return Neo4jGraphStore(
        uri=cfg.neo4j_uri,
        user=cfg.neo4j_user,
        password=cfg.neo4j_password,
    )


def build_retriever(cfg: Settings, store=None, embeddings: EmbeddingProvider | None = None):
    """Construct the hybrid retriever (dense + sparse, RRF-fused)."""
    from app.retrieval.hybrid import HybridRetriever

    store = store or build_vector_store(cfg)
    embeddings = embeddings or build_embeddings(cfg)
    return HybridRetriever(store, embeddings, rrf_k=cfg.rrf_k)


def build_agentic_graph(cfg: Settings, retriever=None, llm: LLMProvider | None = None):
    """
    Construct the compiled CRAG + Self-RAG LangGraph.

    Tavily web fallback is enabled only when TAVILY_API_KEY is present; otherwise the
    loop runs on retrieved papers alone.
    """
    from app.agents.graph import build_graph

    retriever = retriever or build_retriever(cfg)
    llm = llm or build_llm(cfg)
    return build_graph(
        retriever,
        llm,
        max_iterations=cfg.max_iterations,
        crag_threshold=cfg.crag_confidence_threshold,
        tavily_api_key=cfg.tavily_api_key or None,
    )
