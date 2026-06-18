"""
Tests for config startup validation (§0.5).
All tests are deterministic — no API calls or external services.
"""
import pytest
from pydantic import ValidationError


@pytest.mark.unit
class TestConfigValidation:
    def test_missing_required_key_raises(self, monkeypatch):
        """Missing any required secret must raise ValidationError with a clear message."""
        # Wipe all env vars that the Settings class requires
        required = [
            "OPENAI_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY",
            "TAVILY_API_KEY", "NEO4J_PASSWORD",
        ]
        for key in required:
            monkeypatch.delenv(key, raising=False)

        from app.config import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        assert "openai_api_key" in missing_fields

    def test_all_keys_present_succeeds(self, monkeypatch):
        """Providing all required keys must not raise."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GOOGLE_API_KEY", "ggl-test")
        monkeypatch.setenv("COHERE_API_KEY", "co-test")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")

        from app.config import Settings

        cfg = Settings(_env_file=None)
        assert cfg.openai_api_key == "sk-test"

    def test_default_embedding_dimensions(self, monkeypatch):
        """Embedding dimensions default to 3072 (native text-embedding-3-large)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GOOGLE_API_KEY", "ggl-test")
        monkeypatch.setenv("COHERE_API_KEY", "co-test")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")

        from app.config import Settings

        cfg = Settings(_env_file=None)
        assert cfg.embedding_dimensions == 3072

    def test_invalid_chunk_sizes_rejected(self, monkeypatch):
        """child_chunk_size >= parent_chunk_size must raise ValidationError."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GOOGLE_API_KEY", "ggl-test")
        monkeypatch.setenv("COHERE_API_KEY", "co-test")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")
        monkeypatch.setenv("CHILD_CHUNK_SIZE", "2000")
        monkeypatch.setenv("PARENT_CHUNK_SIZE", "1200")

        from app.config import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)
        assert "child_chunk_size" in str(exc_info.value)
