"""FastAPI dependency providers."""

from __future__ import annotations

from collections.abc import Generator

from sqlmodel import Session

from src.db.engine import get_sync_engine


def get_db() -> Generator[Session, None, None]:
    """Yield a synchronous SQLModel session, closing it after the request."""
    with Session(get_sync_engine()) as session:
        yield session
