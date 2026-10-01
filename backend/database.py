# backend/database.py
"""
Provide the database layer for the Intervention Agent audit trail.
Use PostgreSQL in deployed environments and SQLite as a local fallback.
"""

from __future__ import annotations
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

# ============================================================
# Database configuration
# ============================================================

SQLITE_FALLBACK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "interventions.db",
)

DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or f"sqlite:///{SQLITE_FALLBACK_PATH}"
)

IS_SQLITE_FALLBACK = not os.environ.get("DATABASE_URL")


# SQLite needs this flag for use across FastAPI's per-request sessions.
# PostgreSQL doesn't accept it and doesn't need it.
_connect_args = (
    {"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {}
)


engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)


# ============================================================
# SQLAlchemy Base
# ============================================================

class Base(DeclarativeBase):
    """Shared declarative base for every table in backend/models_db.py."""
    pass


# ============================================================
# Database initialization
# ============================================================

def _add_missing_columns() -> None:
    """create_all() never alters an existing table, so add any missing
    nullable columns in place (works on SQLite and PostgreSQL)."""
    insp = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        existing = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing or not col.nullable:
                continue
            col_type = col.type.compile(dialect=engine.dialect)
            with engine.begin() as conn:
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {col_type}'))


def init_db() -> None:
    """Create every missing table, then add any missing nullable columns."""
    from . import models_db  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


# ============================================================
# Session factory
# ============================================================

def get_session():
    """Return a SQLAlchemy session.

    Callers should close it when finished, preferably with:

        with get_session() as session:
            ...
    """
    return SessionLocal()