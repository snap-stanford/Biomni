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
        """Embed and store a summary, keyed by `memory_id`.

        ``metadata`` must carry a non-empty ``user_id``. An episode without an
        owner could otherwise be surfaced to any caller, so we refuse to write
        one (the vector layer never holds an un-isolated memory).
        """
        meta = dict(metadata or {})
        user_id = meta.get("user_id")
        if not user_id or not str(user_id).strip():
            raise ValueError(
                "store_summary requires a non-empty 'user_id' in metadata; "
                "refusing to create an un-isolated vector memory"
            )
        embedding = self.embedding.embed_documents([summary])[0]
        meta.setdefault("memory_id", str(memory_id))
        self.vector_store.add(
            ids=[str(memory_id)],
            texts=[summary],
            embeddings=[embedding],
            metadatas=[meta],
        )
        logger.debug("Stored episodic memory %s", memory_id)

    def search_memory(self, query: str, user_id: str, k: int = 5) -> list[SearchResult]:
        """Return the `k` most similar stored summaries belonging to `user_id`.

        ``user_id`` is enforced as a metadata filter on the vector backend, so a
        caller can never see another user's episodes — even on a similar query.
        """
        query_embedding = self.embedding.embed_query(query)
        return self.vector_store.search(query_embedding, k=k, where={"user_id": user_id})

    def retrieve_memory(self, memory_id: str) -> SearchResult | None:
        """Fetch a single stored summary by id, or None."""
        results = self.vector_store.get([str(memory_id)])
        return results[0] if results else None

    def delete_memory(self, memory_id: str) -> None:
        self.vector_store.delete([str(memory_id)])
