"""
Tests for config startup validation.
All tests are deterministic — no API calls or external services.
"""
import pytest
from pydantic import ValidationError


def _clear_keys(monkeypatch):
    for key in (
        "GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
        "COHERE_API_KEY", "TAVILY_API_KEY", "LLM_PROVIDER",
        "EMBEDDING_PROVIDER", "EMBEDDING_DIMENSIONS",
        "CHILD_CHUNK_SIZE", "PARENT_CHUNK_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.unit
class TestConfigValidation:
    def test_boots_without_any_api_keys(self, monkeypatch):
        """Free-tier default: the app must construct settings with no keys set."""
        _clear_keys(monkeypatch)
        from app.config import Settings

        cfg = Settings(_env_file=None)
        assert cfg.llm_provider == "groq"
        assert cfg.embedding_provider == "fastembed"

    def test_default_embedding_dimensions(self, monkeypatch):
        """FastEmbed default model is 384-dimensional."""
        _clear_keys(monkeypatch)
        from app.config import Settings

        cfg = Settings(_env_file=None)
        assert cfg.embedding_dimensions == 384

    def test_active_llm_key_raises_when_missing(self, monkeypatch):
        """Requesting the active provider key without it set must raise a clear error."""
        _clear_keys(monkeypatch)
        from app.config import Settings

        cfg = Settings(_env_file=None)
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            cfg.active_llm_key()

    def test_active_llm_key_returns_when_present(self, monkeypatch):
        _clear_keys(monkeypatch)
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        from app.config import Settings

        cfg = Settings(_env_file=None)
        assert cfg.active_llm_key() == "gsk-test"

    def test_invalid_chunk_sizes_rejected(self, monkeypatch):
        """child_chunk_size >= parent_chunk_size must raise ValidationError."""
        _clear_keys(monkeypatch)
        monkeypatch.setenv("CHILD_CHUNK_SIZE", "2000")
        monkeypatch.setenv("PARENT_CHUNK_SIZE", "1200")
        from app.config import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)
        assert "child_chunk_size" in str(exc_info.value)
