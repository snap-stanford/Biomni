"""Episodic memory: stores summaries of completed tasks, retrievable by similarity.

Summaries are embedded and stored in a vector backend. Each stored summary is
tagged with the `memory_id` produced by the relational store so that the two
layers can be joined during retrieval.
"""
from __future__ import annotations

import logging
from typing import Sequence

from .vector import EmbeddingProvider, SearchResult, VectorStore

logger = logging.getLogger(__name__)


class EpisodicMemoryStore:
    """Vector-backed store for task summaries."""

    def __init__(self, vector_store: VectorStore, embedding: EmbeddingProvider) -> None:
        self.vector_store = vector_store
        self.embedding = embedding

    def store_summary(
        self,
        memory_id: str,
        summary: str,
        metadata: dict | None = None,
    ) -> None:
        """Embed and store a summary, keyed by `memory_id`."""
        embedding = self.embedding.embed_documents([summary])[0]
        meta = dict(metadata or {})
        meta.setdefault("memory_id", str(memory_id))
        self.vector_store.add(
            ids=[str(memory_id)],
            texts=[summary],
            embeddings=[embedding],
            metadatas=[meta],
        )
        logger.debug("Stored episodic memory %s", memory_id)

    def search_memory(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the `k` most similar stored summaries."""
        query_embedding = self.embedding.embed_query(query)
        return self.vector_store.search(query_embedding, k=k)

    def retrieve_memory(self, memory_id: str) -> SearchResult | None:
        """Fetch a single stored summary by id, or None."""
        results = self.vector_store.get([str(memory_id)])
        return results[0] if results else None

    def delete_memory(self, memory_id: str) -> None:
        self.vector_store.delete([str(memory_id)])
