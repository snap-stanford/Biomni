"""Working memory: the current task state.

Two backends are provided:
  * :class:`SQLWorkingMemoryStore` — persists :class:`WorkingMemoryState` to SQL
    (works on both SQLite and PostgreSQL via SQLAlchemy).
  * :class:`LangGraphCheckpointerStore` — adapts a LangGraph checkpointer so the
    agent's message graph can be checkpointed alongside working memory.

The manager API (`save_state` / `load_state` / `update_state`) is backend-agnostic.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol

from sqlalchemy import DateTime, String, Text, Uuid, delete, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .models import WorkingMemoryState

logger = logging.getLogger(__name__)


class _WorkingBase(DeclarativeBase):
    pass


class WorkingMemoryRow(_WorkingBase):
    __tablename__ = "working_memory"

    task_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    current_step: Mapped[str] = mapped_column(Text, nullable=False, default="")
    variables: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    next_action: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class WorkingMemoryStore(Protocol):
    """Backend interface for working memory."""

    def save(self, state: WorkingMemoryState) -> None: ...

    def load(self, task_id: str) -> WorkingMemoryState | None: ...

    def delete(self, task_id: str) -> None: ...


class SQLWorkingMemoryStore:
    """SQLAlchemy-backed working memory store."""

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory
        # Ensure the table exists (idempotent).
        _WorkingBase.metadata.create_all(self.session_factory.kw["bind"])

    def save(self, state: WorkingMemoryState) -> None:
        import json

        with self.session_factory() as session:
            row = session.get(WorkingMemoryRow, state.task_id)
            payload = {
                "current_step": state.current_step,
                "variables": json.dumps(state.variables, ensure_ascii=False),
                "next_action": state.next_action,
            }
            if row is None:
                session.add(WorkingMemoryRow(task_id=state.task_id, **payload))
            else:
                for k, v in payload.items():
                    setattr(row, k, v)
            session.commit()

    def load(self, task_id: str) -> WorkingMemoryState | None:
        import json

        with self.session_factory() as session:
            row = session.get(WorkingMemoryRow, task_id)
            if row is None:
                return None
            return WorkingMemoryState(
                task_id=row.task_id,
                current_step=row.current_step,
                variables=json.loads(row.variables or "{}"),
                next_action=row.next_action,
            )

    def delete(self, task_id: str) -> None:
        with self.session_factory() as session:
            session.execute(delete(WorkingMemoryRow).where(WorkingMemoryRow.task_id == task_id))
            session.commit()


class LangGraphCheckpointerStore:
    """Adapts a LangGraph checkpointer for working-memory state.

    Uses `task_id` as the checkpoint thread_id. `save` stores an arbitrary
    serializable value; `load` restores it. This is intentionally minimal — a
    full LangGraph integration should checkpoint `AgentState` directly.
    """

    def __init__(self, checkpointer) -> None:
        self.checkpointer = checkpointer

    def save(self, state: WorkingMemoryState) -> None:
        config = {"configurable": {"thread_id": state.task_id}}
        self.checkpointer.put(config, {"state": state.model_dump()})

    def load(self, task_id: str) -> WorkingMemoryState | None:
        config = {"configurable": {"thread_id": task_id}}
        checkpoint = self.checkpointer.get(config)
        if checkpoint is None:
            return None
        data = checkpoint.get("channel_values", {}).get("state")
        if data is None:
            return None
        return WorkingMemoryState(**data)

    def delete(self, task_id: str) -> None:
        # LangGraph checkpointer has no delete; raise to make the limitation explicit.
        raise NotImplementedError("LangGraph checkpointer does not support deletion")


class WorkingMemoryManager:
    """High-level working memory API over a pluggable backend."""

    def __init__(self, store: WorkingMemoryStore) -> None:
        self.store = store

    def save_state(self, state: WorkingMemoryState) -> None:
        self.store.save(state)

    def load_state(self, task_id: str) -> WorkingMemoryState | None:
        return self.store.load(task_id)

    def update_state(self, task_id: str, **updates) -> WorkingMemoryState | None:
        """Load, apply partial updates, and persist. Returns the updated state."""
        state = self.store.load(task_id)
        if state is None:
            state = WorkingMemoryState(task_id=task_id)
        for key, value in updates.items():
            if key in {"current_step", "next_action", "variables", "task_id"}:
                setattr(state, key, value)
        self.store.save(state)
        return state
