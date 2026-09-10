"""MemorySystem: the facade that wires extractor, validator, stores and retriever.

This is the single object the agent holds onto. It hides the full pipeline
behind two methods:
  * :meth:`ingest`  — trace -> extract -> validate -> persist (episodic + semantic)
  * :meth:`retrieve` — query -> search -> assemble -> prompt fragment
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel

from database.migrations import get_engine, get_session_factory, migrate

from .episodic import EpisodicMemoryStore
from .extractor import MemoryExtractor
from .models import MemoryConfig, MemoryExtraction, TraceMessage
from .retriever import MemoryRetriever
from .semantic import SemanticMemoryStore
from .validator import FactValidator
from .vector import build_embedding_provider, build_vector_store
from .working import (
    LangGraphCheckpointerStore,
    SQLWorkingMemoryStore,
    WorkingMemoryManager,
)

logger = logging.getLogger(__name__)


class MemorySystem:
    """Facade for the whole memory pipeline."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        llm: BaseChatModel | None = None,
        checkpointer=None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.llm = llm

        engine = get_engine(self.config.database_url)
        migrate(engine)
        session_factory = get_session_factory(engine)

        vector_store = build_vector_store(self.config)
        embedding = build_embedding_provider(self.config)

        self.validator = FactValidator(min_confidence=self.config.min_confidence)
        self.extractor = MemoryExtractor(llm=llm)
        self.episodic = EpisodicMemoryStore(vector_store, embedding)
        self.semantic = SemanticMemoryStore(
            session_factory,
            self.validator,
            feedback_retract_threshold=self.config.feedback_retract_threshold,
        )
        # Working memory backend: a LangGraph checkpointer when the agent provides
        # one (so current-task state is checkpointed alongside the message graph),
        # otherwise the SQL store (with TTL). Both expose the same manager API.
        if checkpointer is not None:
            working_store = LangGraphCheckpointerStore(checkpointer)
        else:
            working_store = SQLWorkingMemoryStore(
                session_factory, ttl_days=self.config.working_memory_ttl_days
            )
        self.working = WorkingMemoryManager(working_store)
        self.retriever = MemoryRetriever(
            self.episodic,
            self.semantic,
            top_k=self.config.top_k,
            scoring_threshold=self.config.scoring_threshold,
            max_facts=self.config.max_facts,
            confidence_weight=self.config.confidence_weight,
            usage_weight=self.config.usage_weight,
            recency_weight=self.config.recency_weight,
            feedback_weight=self.config.feedback_weight,
            usage_saturation=self.config.usage_saturation,
            recency_lambda=self.config.recency_lambda,
            embedding=embedding,
            fact_similarity_threshold=self.config.fact_similarity_threshold,
        )

    # ---- ingestion -------------------------------------------------------
    async def ingest(
        self,
        trace: Sequence[TraceMessage],
        user_id: str = "default",
        task_id: str | None = None,
    ) -> MemoryExtraction | None:
        """Extract and persist memory from a trace. Never raises into the caller."""
        if not self.config.enabled:
            return None
        if not user_id or not str(user_id).strip():
            logger.error("ingest requires a non-empty user_id; refusing to persist")
            return None
        try:
            extraction = await self.extractor.extract_async(trace)
            memory_id = self.semantic.create_memory(user_id, extraction.summary)
            self.semantic.save_facts(memory_id, extraction.facts)
            self.episodic.store_summary(
                str(memory_id),
                extraction.summary,
                metadata={"user_id": user_id, "task_id": task_id or ""},
            )
            logger.info("Persisted memory %s for user %s", memory_id, user_id)
            return extraction
        except Exception:  # memory must never break the main agent flow
            logger.exception("Memory ingestion failed")
            return None

    def ingest_sync(
        self,
        trace: Sequence[TraceMessage],
        user_id: str = "default",
        task_id: str | None = None,
    ) -> MemoryExtraction | None:
        """Blocking wrapper around :meth:`ingest` for non-async callers."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ingest(trace, user_id, task_id))
        # Already inside an event loop: run without blocking it.
        return asyncio.run_coroutine_threadsafe(
            self.ingest(trace, user_id, task_id), loop
        ).result()

    def ingest_background(
        self,
        trace: Sequence[TraceMessage],
        user_id: str = "default",
        task_id: str | None = None,
    ) -> None:
        """Fire-and-forget ingestion; never blocks the caller.

        If an event loop is running we schedule a task on it; otherwise we spawn
        a daemon thread so synchronous callers are not blocked by extraction.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            import threading

            def _run() -> None:
                try:
                    asyncio.run(self.ingest(trace, user_id, task_id))
                except Exception:
                    logger.exception("Background memory ingestion failed")

            threading.Thread(target=_run, daemon=True, name="memory-ingest").start()
            return
        loop.create_task(self.ingest(trace, user_id, task_id))

    # ---- retrieval -------------------------------------------------------
    def retrieve(self, query: str, user_id: str) -> str:
        """Return a prompt fragment describing relevant past work for ``user_id``."""
        return self.retriever.build_context(query, user_id)

    # ---- cleanup ---------------------------------------------------------
    def cleanup(self, now: datetime | None = None) -> dict:
        """Expire stale facts, then delete fully-invalidated memories.

        A fact's TTL expiry is independent of its episode: only a memory whose
        *every* fact is inactive (expired/superseded/retracted) is deleted, and
        its vector summary removed. A partially-active memory is kept.

        Runs as a unified, retryable process (independent of retrieval, so the
        hot path never rewrites rows). A single ``memory_id`` keys both the SQL
        layer (``Memory`` + ``Fact``) and the vector summary.

        The vector summary is deleted *before* the SQL rows: if the vector delete
        fails we abort before touching SQL, so the ``memory_id`` stays discoverable
        for a later retry; if the SQL delete fails the vector entry is already
        gone (delete is idempotent), so a later retry still works. Cleanup is thus
        idempotent.

        Because SQL and the vector DB are not one transaction, every failure is
        logged with its ``memory_id`` — nothing is silently swallowed.
        """
        now = now or datetime.now(timezone.utc)
        ttl = self.config.fact_ttl_days

        report: dict = {"expired_facts": 0, "deleted": [], "failed": []}

        # 1. Mark stale facts expired (lifecycle transition, based on created_at).
        report["expired_facts"] = self.semantic.expire_facts(now=now, ttl_days=ttl)

        # 2. Delete only memories that no longer have any active fact.
        for memory_id in self.semantic.list_memories_without_active_facts():
            sid = str(memory_id)
            try:
                self.episodic.delete_memory(sid)
            except Exception:
                logger.error(
                    "cleanup: vector delete failed for memory_id=%s", sid, exc_info=True
                )
                report["failed"].append(sid)
                continue
            try:
                self.semantic.delete_memory(memory_id)
            except Exception:
                logger.error(
                    "cleanup: SQL delete failed for memory_id=%s", sid, exc_info=True
                )
                report["failed"].append(sid)
                continue
            logger.info("cleanup: deleted memory_id=%s", sid)
            report["deleted"].append(sid)

        return report
