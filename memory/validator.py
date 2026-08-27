"""Fact validation before facts are persisted.

A fact extracted by the LLM is only persisted if it passes these rules:
  1. confidence >= min_confidence
  2. entity and value are non-empty
  3. it has a source, and the source is not a pure LLM guess (unless allowed)

This keeps low-value and hallucinated "facts" out of the durable store.
"""
from __future__ import annotations

import logging

from .models import MemoryFact

logger = logging.getLogger(__name__)


class FactValidator:
    """Validates extracted facts against confidence/source rules."""

    def __init__(
        self,
        min_confidence: float = 0.6,
        *,
        require_source: bool = True,
        rejected_sources: set[str] | None = None,
    ) -> None:
        self.min_confidence = min_confidence
        self.require_source = require_source
        # Default: reject facts the model merely guessed.
        self.rejected_sources = rejected_sources if rejected_sources is not None else {"llm"}

    def validate_fact(self, fact: MemoryFact) -> bool:
        """Return True if the fact should be stored."""
        if fact.confidence < self.min_confidence:
            return False
        if not fact.entity.strip() or not fact.value.strip():
            return False
        source = (fact.source or "").strip().lower()
        if self.require_source and not source:
            return False
        if source in self.rejected_sources:
            return False
        return True

    def validate(self, facts: list[MemoryFact]) -> list[MemoryFact]:
        """Filter and de-duplicate facts, preserving order."""
        seen: set[tuple[str, str, str]] = set()
        kept: list[MemoryFact] = []
        for fact in facts:
            if not self.validate_fact(fact):
                logger.debug("Rejected fact: %s %s %s", fact.entity, fact.relation, fact.value)
                continue
            key = (fact.entity.strip().lower(), fact.relation.strip().lower(), fact.value.strip())
            if key in seen:
                continue
            seen.add(key)
            kept.append(fact)
        return kept
