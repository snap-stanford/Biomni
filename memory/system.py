"""MemorySystem: the facade that wires extractor, validator, stores and retriever.

This is the single object the agent holds onto. It hides the full pipeline
behind two methods:
  * :meth:`ingest`  — trace -> extract -> validate -> persist (episodic + semantic)
  * :meth:`retrieve` — query -> search -> assemble -> prompt fragment
"""
from __future__ import annotations

import asyncio
import logging
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
from .working import SQLWorkingMemoryStore, WorkingMemoryManager

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
        self.semantic = SemanticMemoryStore(session_factory, self.validator)
        self.working = WorkingMemoryManager(SQLWorkingMemoryStore(session_factory))
        self.retriever = MemoryRetriever(
            self.episodic, self.semantic, top_k=self.config.top_k
        )
        self._checkpointer = checkpointer

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
    def retrieve(self, query: str) -> str:
        """Return a prompt fragment describing relevant past work."""
        return self.retriever.build_context(query)
