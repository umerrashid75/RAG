"""
LLMProvider implementations: Gemini (primary) and OpenAI (fallback/eval).
Both satisfy the LLMProvider Protocol from app.models.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from app.models import LLMProvider

log = logging.getLogger(__name__)


class GeminiProvider:
    """Google Gemini 1.5 Pro provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("google-generativeai is required for GeminiProvider") from exc

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

    def generate(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        response = self._model.generate_content(prompt)
        raw = response.text

        if schema is None:
            return raw

        try:
            # Gemini may wrap JSON in markdown fences
            clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
            return schema.model_validate_json(clean)
        except Exception as exc:
            raise ValueError(f"LLM output did not match schema {schema.__name__}: {exc}") from exc


class OpenAIProvider:
    """OpenAI GPT-4o provider (fallback / eval)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("openai is required for OpenAIProvider") from exc

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        kwargs: dict[str, Any] = {}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **kwargs,
        )
        raw = response.choices[0].message.content or ""

        if schema is None:
            return raw

        try:
            return schema.model_validate_json(raw)
        except Exception as exc:
            raise ValueError(f"LLM output did not match schema {schema.__name__}: {exc}") from exc


class MockLLMProvider:
    """Deterministic stub for unit tests — returns a fixed string or schema instance."""

    def __init__(self, response: str = '{"answer": "mock answer"}') -> None:
        self._response = response

    def generate(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        if schema is None:
            return self._response
        try:
            return schema.model_validate_json(self._response)
        except Exception:
            return schema.model_validate({})


assert isinstance(MockLLMProvider(), LLMProvider)
