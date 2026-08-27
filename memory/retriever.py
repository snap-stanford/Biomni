"""Memory retrieval: turn a user query into an injectable context fragment.

Flow: query -> embed -> vector search (episodic) -> join facts (semantic) via
`memory_id` -> assemble a :class:`MemoryContext` -> render to prompt text.
"""
from __future__ import annotations

import logging

from .episodic import EpisodicMemoryStore
from .models import MemoryContext, MemoryFact
from .semantic import SemanticMemoryStore

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retrieves and assembles relevant memories for a query."""

    def __init__(
        self,
        episodic: EpisodicMemoryStore,
        semantic: SemanticMemoryStore,
        top_k: int = 5,
    ) -> None:
        self.episodic = episodic
        self.semantic = semantic
        self.top_k = top_k

    def retrieve(self, query: str) -> MemoryContext:
        """Retrieve relevant episodic summaries and their facts."""
        context = MemoryContext()
        hits = self.episodic.search_memory(query, k=self.top_k)
        for hit in hits:
            summary = hit.text or hit.metadata.get("summary", "")
            if summary and summary not in context.previous_tasks:
                context.previous_tasks.append(summary)
            memory_id = hit.metadata.get("memory_id")
            if memory_id:
                for fact_row in self.semantic.get_facts_by_memory(memory_id):
                    context.facts.append(
                        MemoryFact(
                            entity=fact_row["entity"],
                            relation=fact_row["relation"],
                            value=fact_row["value"],
                            confidence=fact_row["confidence"],
                            source=fact_row.get("source") or "",
                        )
                    )
        return context

    def build_context(self, query: str) -> str:
        """Return a ready-to-inject prompt fragment (empty string if nothing found)."""
        context = self.retrieve(query)
        return context.to_prompt() if not context.is_empty() else ""
