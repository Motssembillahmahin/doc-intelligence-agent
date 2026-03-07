"""Summarization service — per-document map-reduce and cross-document synthesis.

Entry points:
    summarize_document()  — generate and persist a summary for one document.
    summarize_documents() — synthesize summaries across multiple documents.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlmodel import Session, select

from src.models.database import Chunk, Document
from src.summarization.cross_doc import summarize_across_documents
from src.summarization.map_reduce import summarize_document_chunks

logger = structlog.get_logger(__name__)


def summarize_document(doc_id: uuid.UUID, session: Session) -> str:
    """Run map-reduce summarization for a document and persist the result.

    Loads all chunks ordered by chunk_index, runs the map-reduce pipeline,
    then writes the summary to Document.summary in the DB.

    Args:
        doc_id: UUID of the document to summarize.
        session: Synchronous DB session (caller owns lifecycle).

    Returns:
        The generated summary string (also stored on the document row).

    Raises:
        ValueError: If the document is not found.
    """
    log = logger.bind(doc_id=str(doc_id))

    doc = session.get(Document, doc_id)
    if doc is None:
        raise ValueError(f"Document {doc_id} not found")

    chunks = session.exec(
        select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index)
    ).all()

    log.info("summarize_document_started", chunks=len(chunks))

    summary = summarize_document_chunks(list(chunks), str(doc_id))

    doc.summary = summary
    doc.updated_at = datetime.now(tz=UTC)
    session.commit()

    log.info("summarize_document_persisted", summary_length=len(summary))
    return summary


def summarize_documents(doc_ids: list[uuid.UUID], session: Session) -> str:
    """Produce a cross-document synthesis from multiple document summaries.

    Uses each document's stored summary when available. If a document has no
    summary yet, it is generated on the fly and persisted before synthesis.

    Args:
        doc_ids: List of document UUIDs to include in the synthesis.
        session: Synchronous DB session.

    Returns:
        Cross-document synthesis string, or "" if no valid documents found.
    """
    log = logger.bind(doc_count=len(doc_ids))
    log.info("cross_doc_summarization_started")

    doc_summaries: list[tuple[str, str]] = []

    for doc_id in doc_ids:
        doc = session.get(Document, doc_id)
        if doc is None:
            log.warning("document_not_found", doc_id=str(doc_id))
            continue

        summary = doc.summary
        if not summary:
            log.info("generating_missing_summary", doc_id=str(doc_id))
            summary = summarize_document(doc_id, session)

        if summary:
            doc_summaries.append((doc.filename, summary))

    if not doc_summaries:
        return ""

    return summarize_across_documents(doc_summaries)
