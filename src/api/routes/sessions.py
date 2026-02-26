"""Session routes — create, list, get, delete conversation sessions."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from src.api.deps import get_db
from src.models.database import Message, QueryTrace
from src.models.database import Session as ConvSession
from src.models.schemas import CreateSessionRequest, SessionListResponse, SessionResponse
from src.services.session_service import create_session, get_session, list_sessions

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _to_response(s: ConvSession) -> SessionResponse:
    return SessionResponse(id=s.id, title=s.title, created_at=s.created_at, updated_at=s.updated_at)


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session_endpoint(
    body: CreateSessionRequest,
    db: Annotated[Session, Depends(get_db)],
) -> SessionResponse:
    """Create a new conversation session."""
    session = create_session(db, title=body.title)
    return _to_response(session)


@router.get("", response_model=SessionListResponse)
def list_sessions_endpoint(
    db: Annotated[Session, Depends(get_db)],
    limit: int = 50,
) -> SessionListResponse:
    """List sessions ordered by most-recently updated."""
    sessions = list_sessions(db, limit=limit)
    return SessionListResponse(sessions=[_to_response(s) for s in sessions], total=len(sessions))


@router.get("/{session_id}", response_model=SessionResponse)
def get_session_endpoint(
    session_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> SessionResponse:
    """Get a session by ID."""
    try:
        session = get_session(session_id, db)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found") from None
    return _to_response(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session_endpoint(
    session_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete a session and all its messages and query traces."""
    conv_session = db.get(ConvSession, session_id)
    if conv_session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Delete dependent rows explicitly (passive_deletes=True relies on DB cascade)
    db.exec(select(QueryTrace).where(QueryTrace.session_id == session_id))  # type: ignore[call-overload]
    for qt in db.exec(select(QueryTrace).where(QueryTrace.session_id == session_id)).all():
        db.delete(qt)
    for msg in db.exec(select(Message).where(Message.session_id == session_id)).all():
        db.delete(msg)

    db.delete(conv_session)
    db.commit()
