"""Lightweight schema management for the memory database.

Uses SQLAlchemy `create_all` which is idempotent and sufficient for this scope.
For production PostgreSQL deployments, swap this module for Alembic — the ORM
models in :mod:`database.models` remain the single source of truth.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
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
    """Create all missing tables, then add any newly-introduced columns.

    ``create_all`` is idempotent but will NOT add columns to tables that already
    exist, so this also runs a lightweight column migration for the fact
    lifecycle/scoring columns. Safe to call repeatedly.
    """
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    logger.info("Memory schema is up to date")


def _add_missing_columns(engine: Engine) -> None:
    """Add fact lifecycle/scoring columns to an existing ``facts`` table.

    Only touches columns that are genuinely missing, so re-running is a no-op.
    SQLite and PostgreSQL get dialect-appropriate column types/defaults.
    """
    inspector = inspect(engine)
    if "facts" not in inspector.get_table_names():
        # Fresh database: create_all already produced the full schema.
        return

    existing = {col["name"] for col in inspector.get_columns("facts")}

    if engine.dialect.name == "sqlite":
        column_specs = {
            "created_at": ("DATETIME", "CURRENT_TIMESTAMP"),
            "updated_at": ("DATETIME", "CURRENT_TIMESTAMP"),
            "status": ("VARCHAR(255)", "'active'"),
            "access_count": ("INTEGER", "0"),
            "positive_feedback_count": ("INTEGER", "0"),
            "negative_feedback_count": ("INTEGER", "0"),
        }
    else:
        column_specs = {
            "created_at": ("TIMESTAMP WITH TIME ZONE", "now()"),
            "updated_at": ("TIMESTAMP WITH TIME ZONE", "now()"),
            "status": ("VARCHAR(255)", "'active'"),
            "access_count": ("INTEGER", "0"),
            "positive_feedback_count": ("INTEGER", "0"),
            "negative_feedback_count": ("INTEGER", "0"),
        }

    for column, (col_type, default) in column_specs.items():
        if column in existing:
            continue
        ddl = f"ALTER TABLE facts ADD COLUMN {column} {col_type} NOT NULL DEFAULT {default}"
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("Migrated facts table: added column %s", column)
