"""
LLM entity and relationship extraction from research-paper text.
Output is validated against Pydantic schema; malformed extractions are rejected.
"""
from __future__ import annotations

import logging

from app.models import ExtractionResult, LLMProvider

log = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
Extract the entities and their relationships from the following AI research-paper text.

Entity types: Paper, Author, Method, Dataset, Benchmark, Task, Concept, Institution
Relation types: CITES, PROPOSES, EVALUATES_ON, BUILDS_ON, USES, AUTHORED_BY, \
OUTPERFORMS, INTRODUCES, AFFILIATED_WITH

Text:
{text}

Respond in strict JSON:
{{
  "entities": [{{"name": "...", "type": "...", "attributes": {{}}}}],
  "relations": [{{"src": "...", "relation": "...", "dst": "...", "attributes": {{}}}}]
}}
"""


def extract_entities(
    text: str,
    llm: LLMProvider,
    *,
    max_text_length: int = 3000,
) -> ExtractionResult:
    """
    Extract entities and relations from *text* using *llm*.
    Raises ValueError if the LLM output fails schema validation.
    """
    truncated = text[:max_text_length]
    prompt = _EXTRACTION_PROMPT.format(text=truncated)

    result = llm.generate(prompt, schema=ExtractionResult)
    if not isinstance(result, ExtractionResult):
        raise ValueError(f"Extraction returned unexpected type: {type(result)}")
    return result
