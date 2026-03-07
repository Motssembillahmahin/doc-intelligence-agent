"""Session service — thin orchestration layer for conversation session management."""

from __future__ import annotations

import uuid

import structlog
from sqlmodel import Session as DBSession

from src.memory.session_manager import (
    create_session as _create,
)
from src.memory.session_manager import (
    get_session as _get,
)
from src.memory.session_manager import (
    list_sessions as _list,
)
from src.models.database import Session

logger = structlog.get_logger(__name__)


def create_session(db_session: DBSession, title: str | None = None) -> Session:
    """Create and persist a new conversation session."""
    return _create(db_session, title=title)


def get_session(session_id: uuid.UUID, db_session: DBSession) -> Session:
    """Return a Session by ID.

    Raises:
        ValueError: If the session does not exist.
    """
    session = _get(session_id, db_session)
    if session is None:
        raise ValueError(f"Session {session_id} not found")
    return session


def list_sessions(db_session: DBSession, limit: int = 50) -> list[Session]:
    """Return sessions ordered by most-recently updated."""
    return _list(db_session, limit=limit)
