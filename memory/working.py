"""Working memory: the current task state.

Two backends are provided:
  * :class:`SQLWorkingMemoryStore` — persists :class:`WorkingMemoryState` to SQL
    (works on both SQLite and PostgreSQL via SQLAlchemy). Supports TTL expiry and
    optimistic locking via a ``version`` column.
  * :class:`LangGraphCheckpointerStore` — adapts a LangGraph checkpointer so the
    agent's message graph can be checkpointed alongside working memory.

The manager API (``save_state`` / ``load_state`` / ``update_state`` plus the
``set_variable`` / ``get_variable`` / ``delete_variable`` helpers and ``clear`` /
``cleanup``) is backend-agnostic.

Working memory is *not* long-term memory: it holds the current task's state
(step, variables, next action) for the duration of one task, then is cleared or
left to expire. Durable facts about past work belong to episodic/semantic memory.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Protocol

from sqlalchemy import DateTime, Integer, String, Text, delete, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .models import WorkingMemoryState

logger = logging.getLogger(__name__)


class _WorkingBase(DeclarativeBase):
    pass


class WorkingMemoryRow(_WorkingBase):
    __tablename__ = "working_memory"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True, default="default")
    task_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    current_step: Mapped[str] = mapped_column(Text, nullable=False, default="")
    variables: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    next_action: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))


class ConcurrentUpdateError(RuntimeError):
    """Raised when a working-memory save detects a version mismatch (a lost update)."""


class WorkingMemoryStore(Protocol):
    """Backend interface for working memory."""

    def save(self, state: WorkingMemoryState) -> None: ...

    def load(self, user_id: str, task_id: str) -> WorkingMemoryState | None: ...

    def delete(self, user_id: str, task_id: str) -> None: ...

    def delete_expired(self, now: datetime) -> int: ...


class SQLWorkingMemoryStore:
    """SQLAlchemy-backed working memory store with TTL and optimistic locking."""

    def __init__(self, session_factory, ttl_days: int | None = None) -> None:
        self.session_factory = session_factory
        self.ttl_days = ttl_days
        # Ensure the table exists (idempotent).
        _WorkingBase.metadata.create_all(self.session_factory.kw["bind"])

    def save(self, state: WorkingMemoryState) -> None:
        now = datetime.now(timezone.utc)
        with self.session_factory() as session:
            row = session.get(WorkingMemoryRow, (state.user_id, state.task_id))
            payload = {
                "current_step": state.current_step,
                "variables": json.dumps(state.variables, ensure_ascii=False),
                "next_action": state.next_action,
            }
            if row is None:
                expires_at = None
                if self.ttl_days is not None:
                    expires_at = now + timedelta(days=self.ttl_days)
                session.add(
                    WorkingMemoryRow(
                        user_id=state.user_id,
                        task_id=state.task_id,
                        expires_at=expires_at,
                        version=state.version,
                        **payload,
                    )
                )
            else:
                if row.version != state.version:
                    raise ConcurrentUpdateError(
                        f"working_memory ({state.user_id!r}, {state.task_id!r}) stored "
                        f"version {row.version} != expected {state.version}; aborting to "
                        f"avoid a lost update"
                    )
                for k, v in payload.items():
                    setattr(row, k, v)
                row.version = row.version + 1
            session.commit()

    def load(self, user_id: str, task_id: str) -> WorkingMemoryState | None:
        with self.session_factory() as session:
            row = session.get(WorkingMemoryRow, (user_id, task_id))
            if row is None:
                return None
            return WorkingMemoryState(
                task_id=row.task_id,
                user_id=row.user_id,
                current_step=row.current_step,
                variables=json.loads(row.variables or "{}"),
                next_action=row.next_action,
                created_at=row.created_at,
                updated_at=row.updated_at,
                expires_at=row.expires_at,
                version=row.version,
            )

    def delete(self, user_id: str, task_id: str) -> None:
        with self.session_factory() as session:
            session.execute(
                delete(WorkingMemoryRow).where(
                    WorkingMemoryRow.user_id == user_id,
                    WorkingMemoryRow.task_id == task_id,
                )
            )
            session.commit()

    def delete_expired(self, now: datetime) -> int:
        """Delete working-memory rows whose TTL has elapsed. Returns the count removed."""
        with self.session_factory() as session:
            result = session.execute(
                delete(WorkingMemoryRow).where(
                    WorkingMemoryRow.expires_at.is_not(None),
                    WorkingMemoryRow.expires_at <= now,
                )
            )
            session.commit()
            return result.rowcount or 0


class LangGraphCheckpointerStore:
    """Adapts a LangGraph checkpointer for working-memory state.

    Uses ``thread_id = "<user_id>:<task_id>"`` so each (user, task) pair has its
    own checkpoint namespace. ``save`` stores a JSON-safe dump of the state under
    ``channel_values["working_memory"]``; ``load`` restores it.

    This is intentionally minimal: a full integration would checkpoint the whole
    ``AgentState`` (including the message graph) rather than a side-channel key.
    Deletion and time-based expiry are not expressible on a LangGraph checkpointer,
    so ``delete`` raises and ``delete_expired`` is a no-op — callers should rely on
    the checkpointer's own retention policy.
    """

    def __init__(self, checkpointer) -> None:
        self.checkpointer = checkpointer

    @staticmethod
    def _thread_id(user_id: str, task_id: str) -> str:
        return f"{user_id}:{task_id}"

    def save(self, state: WorkingMemoryState) -> None:
        config = {"configurable": {"thread_id": self._thread_id(state.user_id, state.task_id)}}
        data = json.loads(state.model_dump_json())
        self.checkpointer.put(config, {"channel_values": {"working_memory": data}})

    def load(self, user_id: str, task_id: str) -> WorkingMemoryState | None:
        config = {"configurable": {"thread_id": self._thread_id(user_id, task_id)}}
        data = self._extract(self.checkpointer.get(config))
        if data is None:
            return None
        return WorkingMemoryState(**data)

    @staticmethod
    def _extract(result):
        if result is None:
            return None
        # A LangGraph MemorySaver returns a checkpoint tuple whose ``.checkpoint``
        # carries ``.channel_values``; a simple dict-based fake returns a plain dict.
        checkpoint = getattr(result, "checkpoint", result)
        if isinstance(checkpoint, dict):
            channel_values = checkpoint.get("channel_values", checkpoint)
        else:
            channel_values = getattr(checkpoint, "channel_values", {})
        if isinstance(channel_values, dict):
            return channel_values.get("working_memory")
        return getattr(channel_values, "working_memory", None)

    def delete(self, user_id: str, task_id: str) -> None:
        # LangGraph checkpointer has no delete; raise to make the limitation explicit.
        raise NotImplementedError("LangGraph checkpointer does not support deletion")

    def delete_expired(self, now: datetime) -> int:
        # A checkpointer cannot enumerate or expire entries by time.
        return 0


class WorkingMemoryManager:
    """High-level working memory API over a pluggable backend.

    Identity is ``(user_id, task_id)``. Both are fixed at creation and are never
    settable through ``update_state`` — the ordinary update path can only change
    ``current_step`` / ``next_action`` / ``variables``.
    """

    def __init__(self, store: WorkingMemoryStore) -> None:
        self.store = store

    def save_state(self, state: WorkingMemoryState) -> None:
        self.store.save(state)

    def load_state(self, user_id: str, task_id: str) -> WorkingMemoryState | None:
        return self.store.load(user_id, task_id)

    def create_state(
        self,
        user_id: str,
        task_id: str,
        current_step: str = "",
        next_action: str = "",
        variables: dict | None = None,
    ) -> WorkingMemoryState:
        """Create a fresh working-memory entry (or overwrite an existing one)."""
        state = WorkingMemoryState(
            user_id=user_id,
            task_id=task_id,
            current_step=current_step,
            next_action=next_action,
            variables=variables or {},
        )
        self.store.save(state)
        return state

    def update_state(self, user_id: str, task_id: str, **updates) -> WorkingMemoryState | None:
        """Load, apply partial updates, and persist. Returns the updated state.

        Only ``current_step``, ``next_action`` and ``variables`` are mutable here;
        ``user_id`` / ``task_id`` are immutable identity and are ignored if passed.
        """
        state = self.store.load(user_id, task_id)
        if state is None:
            state = WorkingMemoryState(user_id=user_id, task_id=task_id)
        for key, value in updates.items():
            if key in {"current_step", "next_action", "variables"}:
                setattr(state, key, value)
        self.store.save(state)
        return state

    # ---- structured variables API ----------------------------------------
    def set_variable(self, user_id: str, task_id: str, key: str, value) -> None:
        """Set a single variable without rewriting the whole ``variables`` blob."""
        state = self.store.load(user_id, task_id)
        if state is None:
            state = WorkingMemoryState(user_id=user_id, task_id=task_id)
        state.variables[key] = value
        self.store.save(state)

    def get_variable(self, user_id: str, task_id: str, key: str, default=None):
        state = self.store.load(user_id, task_id)
        if state is None:
            return default
        return state.variables.get(key, default)

    def delete_variable(self, user_id: str, task_id: str, key: str) -> bool:
        """Remove a variable; returns True if it was present and removed."""
        state = self.store.load(user_id, task_id)
        if state is None or key not in state.variables:
            return False
        del state.variables[key]
        self.store.save(state)
        return True

    # ---- lifecycle -------------------------------------------------------
    def clear(self, user_id: str, task_id: str) -> None:
        """Delete a task's working-memory entry (task finished / abandoned)."""
        self.store.delete(user_id, task_id)

    def cleanup(self, now: datetime | None = None) -> int:
        """Expire working-memory entries past their TTL. Returns the count removed."""
        now = now or datetime.now(timezone.utc)
        return self.store.delete_expired(now)
