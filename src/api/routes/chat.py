"""Chat routes — grounded Q&A with session context."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from src.api.deps import get_db
from src.models.schemas import ChatRequest, ChatResponse
from src.services.chat_service import chat
from src.services.session_service import get_session

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
) -> ChatResponse:
    """Answer a question using RAG over ingested documents.

    Requires a valid session_id. Optionally scoped to specific doc_ids.
    """
    # Validate session exists before entering the chat pipeline
    try:
        get_session(request.session_id, db)
    except ValueError:
        raise HTTPException(  # noqa: B904
            status_code=404, detail=f"Session {request.session_id} not found"
        ) from None

    doc_ids = [str(d) for d in request.doc_ids] if request.doc_ids else None

    return chat(
        query=request.query,
        session_id=request.session_id,
        db_session=db,
        doc_ids=doc_ids,
    )
