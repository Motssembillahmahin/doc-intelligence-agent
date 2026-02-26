"""Session manager — CRUD operations for conversation Session records."""

from __future__ import annotations

import uuid

import structlog
from sqlmodel import Session as DBSession
from sqlmodel import select

from src.models.database import Session

logger = structlog.get_logger(__name__)


def create_session(db_session: DBSession, title: str | None = None) -> Session:
    """Create a new conversation session and persist it."""
    session = Session(title=title)
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    logger.info("session_created", session_id=str(session.id))
    return session


def get_session(session_id: uuid.UUID, db_session: DBSession) -> Session | None:
    """Return a Session by ID, or None if not found."""
    return db_session.get(Session, session_id)


def list_sessions(db_session: DBSession, limit: int = 50) -> list[Session]:
    """Return sessions ordered by most-recently updated, up to limit."""
    return list(
        db_session.exec(select(Session).order_by(Session.updated_at.desc()).limit(limit)).all()
    )
