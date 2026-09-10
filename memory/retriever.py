"""Memory retrieval: turn a user query into an injectable context fragment.

Flow: query -> embed -> vector search (episodic) -> join facts (semantic) via
`memory_id` -> two-stage ranking -> cap to `max_facts` -> assemble a
:class:`MemoryContext` -> render to prompt text.

Vector similarity ranks *episodes*; once the episode's facts are fetched, they
are ranked by fact-level signals (confidence in the cold-start phase, the full
``importance_score`` once mature). The two notions are deliberately kept apart.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .models import MemoryContext, MemoryFact
from .scoring import importance_score

if TYPE_CHECKING:
    from .episodic import EpisodicMemoryStore
    from .semantic import SemanticMemoryStore
    from .vector import EmbeddingProvider

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retrieves and assembles relevant memories for a query."""

    def __init__(
        self,
        episodic: EpisodicMemoryStore,
        semantic: SemanticMemoryStore,
        top_k: int = 5,
        *,
        scoring_threshold: int = 3,
        max_facts: int = 20,
        confidence_weight: float = 0.4,
        usage_weight: float = 0.2,
        recency_weight: float = 0.2,
        feedback_weight: float = 0.2,
        usage_saturation: float = 100.0,
        recency_lambda: float = 0.01,
        embedding: EmbeddingProvider | None = None,
        fact_similarity_threshold: float = 0.45,
        similarity_weight: float = 0.7,
    ) -> None:
        self.episodic = episodic
        self.semantic = semantic
        self.top_k = top_k
        self.scoring_threshold = scoring_threshold
        self.max_facts = max_facts
        self.confidence_weight = confidence_weight
        self.usage_weight = usage_weight
        self.recency_weight = recency_weight
        self.feedback_weight = feedback_weight
        self.usage_saturation = usage_saturation
        self.recency_lambda = recency_lambda
        # Query-aware fact reranking (opt-in via DI). ``None`` keeps the original
        # importance-only ranking so existing callers/tests are unaffected.
        self.embedding = embedding
        # Retained for backward compatibility (``MemoryConfig``/``MemorySystem`` still
        # pass it). The query-aware ranking no longer drops facts below a similarity
        # threshold: semantic similarity only influences the *rank*, never deletion.
        self.fact_similarity_threshold = fact_similarity_threshold
        self.similarity_weight = similarity_weight

    def retrieve(self, query: str, user_id: str) -> MemoryContext:
        """Retrieve relevant episodic summaries and their highest-value facts.

        ``user_id`` is mandatory: the vector search is scoped to this user, and
        the SQL fact lookup re-verifies ownership.
        """
        context = MemoryContext()
        hits = self.episodic.search_memory(query, user_id, k=self.top_k)

        memory_ids: list[str] = []
        for hit in hits:
            summary = hit.text or hit.metadata.get("summary", "")
            if summary and summary not in context.previous_tasks:
                context.previous_tasks.append(summary)
            memory_id = hit.metadata.get("memory_id")
            if memory_id:
                memory_ids.append(memory_id)

        # One batched SQL query for all memories (avoids N+1), then rank globally.
        candidate_rows: list[dict] = self.semantic.get_active_facts_by_memories(memory_ids, user_id)

        selected = self._rank_and_select(candidate_rows, query=query)

        # Only facts that actually enter the context count as "accessed".
        if selected:
            self.semantic.increment_access_counts([row["id"] for row in selected])

        for row in selected:
            context.facts.append(self._to_memory_fact(row))

        return context

    def build_context(self, query: str, user_id: str) -> str:
        """Return a ready-to-inject prompt fragment (empty string if nothing found)."""
        context = self.retrieve(query, user_id)
        return context.to_prompt() if not context.is_empty() else ""

    # ---- ranking ---------------------------------------------------------
    def _rank_and_select(self, rows: list[dict], query: str | None = None) -> list[dict]:
        """Rank candidate facts and return the top ``max_facts``.

        Two modes:

        * **query-aware** (``query`` and ``self.embedding`` set): score each fact by
          ``similarity_weight * cos(query, fact) + (1 - similarity_weight) *
          importance`` (0.7 / 0.3 by default), then keep the top ``max_facts``.
          Semantic similarity only *ranks* — no fact is dropped for falling below a
          threshold.
        * **importance-only** (fallback): the original two-stage rule —
          ``access_count <= scoring_threshold`` ranks by confidence (cold start),
          otherwise by ``importance_score`` (mature).
        """
        if not rows:
            return []
        now = datetime.now(UTC)
        if query is not None and self.embedding is not None:
            scored = self._score_query_aware(rows, query, now)
            scored.sort(key=lambda item: item[0], reverse=True)
            return [item[2] for item in scored[: self.max_facts]]
        ranked = sorted(rows, key=lambda r: self._ranking_score(r, now), reverse=True)
        return ranked[: self.max_facts]

    @staticmethod
    def _fact_text(row: dict) -> str:
        """Render a fact as an embeddable sentence: ``entity relation value``."""
        return f"{row['entity']} {row['relation']} {row['value']}"

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity in [-1, 1]; guards against zero-length vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (norm_a * norm_b)

    def _score_query_aware(self, rows: list[dict], query: str, now: datetime) -> list[tuple[float, float, dict]]:
        """Return ``(score, similarity, row)`` triples for query-aware ranking.

        ``score = similarity_weight * similarity + (1 - similarity_weight) *
        importance`` (0.7 / 0.3 by default). ``similarity`` is the cosine between
        the query embedding and the fact's ``entity relation value`` embedding;
        ``importance`` reuses the existing two-stage signal. Similarity dominates
        but is never a hard filter — a low-similarity fact still enters the ranking
        and is ordered by this combined score.
        """
        query_vec = self.embedding.embed_query(query)
        fact_vecs = self.embedding.embed_documents([self._fact_text(row) for row in rows])
        out: list[tuple[float, float, dict]] = []
        for row, fact_vec in zip(rows, fact_vecs, strict=False):
            similarity = self._cosine_similarity(query_vec, fact_vec)
            importance = self._ranking_score(row, now)
            combined = self.similarity_weight * similarity + (1.0 - self.similarity_weight) * importance
            out.append((combined, similarity, row))
        return out

    def _ranking_score(self, row: dict, now: datetime) -> float:
        """Score used to sort a single candidate fact."""
        access_count = int(row.get("access_count", 0))
        confidence = float(row.get("confidence", 0.0))
        if access_count <= self.scoring_threshold:
            return confidence
        return importance_score(
            confidence=confidence,
            access_count=access_count,
            created_at=row.get("created_at") or now,
            now=now,
            confidence_weight=self.confidence_weight,
            usage_weight=self.usage_weight,
            recency_weight=self.recency_weight,
            feedback_weight=self.feedback_weight,
            positive_feedback=row.get("positive_feedback_count", 0),
            negative_feedback=row.get("negative_feedback_count", 0),
            usage_saturation=self.usage_saturation,
            recency_lambda=self.recency_lambda,
        )

    def _to_memory_fact(self, row: dict) -> MemoryFact:
        """Build a ``MemoryFact``, always attaching the composite importance score."""
        now = datetime.now(UTC)
        created_at = row.get("created_at") or now
        confidence = float(row.get("confidence", 0.0))
        access_count = int(row.get("access_count", 0))
        return MemoryFact(
            entity=row["entity"],
            relation=row["relation"],
            value=row["value"],
            confidence=confidence,
            source=row.get("source") or "",
            created_at=created_at,
            updated_at=row.get("updated_at"),
            status=row.get("status") or "active",
            importance_score=importance_score(
                confidence=confidence,
                access_count=access_count,
                created_at=created_at,
                now=now,
                confidence_weight=self.confidence_weight,
                usage_weight=self.usage_weight,
                recency_weight=self.recency_weight,
                feedback_weight=self.feedback_weight,
                positive_feedback=row.get("positive_feedback_count", 0),
                negative_feedback=row.get("negative_feedback_count", 0),
                usage_saturation=self.usage_saturation,
                recency_lambda=self.recency_lambda,
            ),
        )
