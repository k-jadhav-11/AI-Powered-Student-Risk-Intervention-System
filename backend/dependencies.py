# backend/dependencies.py
"""
Provide FastAPI dependencies for database session management.
Expose the database session through FastAPI dependency injection.
"""
from __future__ import annotations

from typing import Iterator
from sqlalchemy.orm import Session
from .database import SessionLocal


def get_db() -> Iterator[Session]:
    """Provide a request-scoped SQLAlchemy session for FastAPI dependencies."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
