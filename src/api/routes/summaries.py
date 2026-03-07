"""Summary routes — per-document and cross-document summarization."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from src.api.deps import get_db
from src.models.database import Document
from src.models.schemas import CrossDocSummaryRequest, CrossDocSummaryResponse, SummarizeResponse
from src.services.summarization_service import (
    summarize_document,
    summarize_documents,
)

router = APIRouter(prefix="/summaries", tags=["summaries"])


@router.post("/documents/{doc_id}", response_model=SummarizeResponse)
def summarize_document_endpoint(
    doc_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> SummarizeResponse:
    """Generate (or regenerate) a summary for a document via map-reduce."""
    try:
        summary = summarize_document(doc_id, db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SummarizeResponse(doc_id=doc_id, summary=summary)


@router.get("/documents/{doc_id}", response_model=SummarizeResponse)
def get_summary_endpoint(
    doc_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> SummarizeResponse:
    """Return the stored summary for a document (does not re-generate)."""
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    if not doc.summary:
        raise HTTPException(
            status_code=404,
            detail=f"No summary for document {doc_id}. POST to generate one.",
        )
    return SummarizeResponse(doc_id=doc_id, summary=doc.summary)


@router.post("/cross-doc", response_model=CrossDocSummaryResponse)
def cross_doc_summary_endpoint(
    body: CrossDocSummaryRequest,
    db: Annotated[Session, Depends(get_db)],
) -> CrossDocSummaryResponse:
    """Synthesize summaries across multiple documents."""
    if not body.doc_ids:
        raise HTTPException(status_code=422, detail="At least one doc_id is required")
    synthesis = summarize_documents([d for d in body.doc_ids], db)
    if not synthesis:
        raise HTTPException(status_code=404, detail="No summaries could be generated")
    return CrossDocSummaryResponse(synthesis=synthesis)
