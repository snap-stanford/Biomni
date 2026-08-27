"""Lightweight schema management for the memory database.

Uses SQLAlchemy `create_all` which is idempotent and sufficient for this scope.
For production PostgreSQL deployments, swap this module for Alembic — the ORM
models in :mod:`database.models` remain the single source of truth.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)


def get_engine(database_url: str, echo: bool = False) -> Engine:
    """Build a SQLAlchemy engine, handling SQLite path setup transparently."""
    kwargs: dict = {}
    if database_url.startswith("sqlite"):
        path = database_url.removeprefix("sqlite:///")
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, echo=echo, future=True, **kwargs)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a session factory for dependency injection."""
    return sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False, future=True
    )


def migrate(engine: Engine) -> None:
    """Create all missing tables. Safe to call repeatedly."""
    Base.metadata.create_all(engine)
    logger.info("Memory schema is up to date")
