"""SQLAlchemy ORM models for the memory system.

The `memory.id` column is the database-generated primary key used to join the
relational `facts` table with the episodic (vector) store. It is NEVER produced
by the LLM — it is assigned by the database on insert.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all memory tables."""


class Memory(Base):
    """One recorded episode: the summary of a completed agent run."""

    __tablename__ = "memory"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    facts: Mapped[list["Fact"]] = relationship(
        back_populates="memory", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (Index("ix_memory_user_created", "user_id", "created_at"),)


class Fact(Base):
    """One extracted fact, linked to the episode (memory) that produced it."""

    __tablename__ = "facts"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    memory_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("memory.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    relation: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)

    memory: Mapped[Memory] = relationship(back_populates="facts")

    __table_args__ = (Index("ix_facts_entity_relation", "entity", "relation"),)
