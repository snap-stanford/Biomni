"""Vector store abstraction with swappable backends and embedding providers.

This module is the seam that lets the episodic memory swap between Chroma,
FAISS, and Milvus without touching the rest of the system. Only Chroma is
implemented here; FAISS/Milvus implement the same :class:`VectorStore`
protocol and are registered in :func:`build_vector_store`.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single vector-search hit."""

    id: str
    score: float  # higher = more similar
    metadata: dict = field(default_factory=dict)
    text: str = ""


class EmbeddingProvider(Protocol):
    """Anything that can embed documents and queries."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """The minimum surface a vector backend must implement."""

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None: ...

    def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]: ...

    def get(self, ids: list[str]) -> list[SearchResult]: ...

    def delete(self, ids: list[str]) -> None: ...


class HashingEmbedding:
    """Deterministic, dependency-free embedding for tests and offline use.

    NOT semantically meaningful — do not use where recall quality matters.
    Exists so the system runs with zero external embedding dependencies.
    """

    dim: int = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LangChainEmbeddingProvider:
    """Wraps any LangChain `Embeddings` instance (OpenAI, HuggingFace, etc.)."""

    def __init__(self, embeddings) -> None:
        self._embeddings = embeddings

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)


class ChromaVectorStore:
    """Chroma-backed vector store. `chromadb` is imported lazily so it is optional."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "episodic_memory",
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _ensure(self):
        if self._collection is not None:
            return self._collection
        import chromadb  # local import keeps chromadb optional

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        return self._collection

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        col = self._ensure()
        col.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(ids),
        )

    def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]:
        col = self._ensure()
        res = col.query(query_embeddings=[query_embedding], n_results=k)
        results: list[SearchResult] = []
        ids = res.get("ids") or [[]]
        distances = res.get("distances") or [[0.0] * len(ids[0])]
        metadatas = res.get("metadatas") or [[{}] * len(ids[0])]
        documents = res.get("documents") or [[""] * len(ids[0])]
        for i, cid in enumerate(ids[0]):
            # Chroma cosine distance in [0, 2]; convert to similarity (higher = better).
            distance = distances[0][i] if distances[0] else 0.0
            results.append(
                SearchResult(
                    id=cid,
                    score=1.0 - float(distance),
                    metadata=metadatas[0][i] or {},
                    text=documents[0][i] or "",
                )
            )
        return results

    def get(self, ids: list[str]) -> list[SearchResult]:
        col = self._ensure()
        res = col.get(ids=ids)
        out: list[SearchResult] = []
        for i, cid in enumerate(res["ids"]):
            out.append(
                SearchResult(
                    id=cid,
                    score=0.0,
                    metadata=(res.get("metadatas") or [{}])[i] or {},
                    text=(res.get("documents") or [""])[i] or "",
                )
            )
        return out

    def delete(self, ids: list[str]) -> None:
        col = self._ensure()
        col.delete(ids=ids)


def build_vector_store(config) -> VectorStore:
    """Factory for vector backends selected by config.vector_db."""
    kind = (config.vector_db or "chroma").lower()
    if kind == "chroma":
        return ChromaVectorStore(
            persist_dir=config.persist_dir, collection_name=config.collection_name
        )
    if kind == "faiss":
        # FAISS is an index-only store (no metadata); implement by pairing it
        # with a sidecar metadata store. Left as an extension point.
        raise NotImplementedError(
            "FAISS backend not yet implemented; add an index + metadata sidecar."
        )
    if kind == "milvus":
        raise NotImplementedError(
            "Milvus backend not yet implemented; implement the VectorStore protocol."
        )
    raise ValueError(f"Unknown vector_db: {kind}")


def build_embedding_provider(config) -> EmbeddingProvider:
    """Factory for embedding providers selected by config.embedding_provider."""
    kind = (config.embedding_provider or "hash").lower()
    if kind == "hash":
        return HashingEmbedding()
    if kind == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "langchain-openai is required for OpenAI embeddings. "
                "pip install langchain-openai"
            ) from exc
        return LangChainEmbeddingProvider(
            OpenAIEmbeddings(model=config.embedding_model)
        )
    if kind == "langchain":
        raise ValueError(
            "Provide an Embeddings instance directly when embedding_provider='langchain'."
        )
    raise ValueError(f"Unknown embedding_provider: {kind}")
