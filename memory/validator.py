"""Fact validation before facts are persisted.

A fact extracted by the LLM is only persisted if it passes these rules:
  1. confidence >= min_confidence
  2. entity and value are non-empty
  3. it has a source, and the source is not a pure LLM guess (unless allowed)

This keeps low-value and hallucinated "facts" out of the durable store.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    def _rejection_reason(self, fact: MemoryFact) -> str | None:
        """Return why a fact should be rejected, or None if it should be kept."""
        if fact.confidence < self.min_confidence:
            return f"confidence {fact.confidence} < min {self.min_confidence}"
        if not fact.entity.strip():
            return "empty entity"
        if not fact.value.strip():
            return "empty value"
        source = (fact.source or "").strip().lower()
        if self.require_source and not source:
            return "missing source"
        if source in self.rejected_sources:
            return f"rejected source '{source}'"
        return None

    def validate_fact(self, fact: MemoryFact) -> bool:
        """Return True if the fact should be stored."""
        return self._rejection_reason(fact) is None

    def validate(self, facts: list[MemoryFact]) -> list[MemoryFact]:
        """Filter and de-duplicate facts, preserving order."""
        seen: set[tuple[str, str, str]] = set()
        kept: list[MemoryFact] = []
        for fact in facts:
            reason = self._rejection_reason(fact)
            if reason is not None:
                logger.debug(
                    "Rejected fact: %s %s %s (reason: %s)",
                    fact.entity,
                    fact.relation,
                    fact.value,
                    reason,
                )
                continue
            key = (fact.entity.strip().lower(), fact.relation.strip().lower(), fact.value.strip())
            if key in seen:
                continue
            seen.add(key)
            kept.append(fact)
        return kept
