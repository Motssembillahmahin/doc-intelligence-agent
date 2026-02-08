"""Ingestion service — orchestrates PDF ingestion with DB tracking."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import structlog

from src.db.engine import get_sync_session
from src.ingestion.pipeline import IngestionResult, run_ingestion
from src.models.database import Document
from src.models.enums import DocumentStatus

logger = structlog.get_logger(__name__)


def ingest_document(doc_id: uuid.UUID, file_path: Path) -> IngestionResult:
    """Run ingestion pipeline and update document status in DB.

    This is the main entry point called by the Celery task.
    Uses sync DB session since Celery workers are synchronous.
    """
    log = logger.bind(doc_id=str(doc_id))
    session = get_sync_session()

    try:
        # Mark as processing
        doc = session.get(Document, doc_id)
        if doc is None:
            log.error("document_not_found")
            raise ValueError(f"Document {doc_id} not found")

        doc.status = DocumentStatus.processing
        doc.updated_at = datetime.now(tz=UTC)
        session.commit()

        log.info("ingestion_started")
        result = run_ingestion(file_path, str(doc_id))

        if result.success:
            doc.page_count = result.validation.page_count
            if result.warnings:
                doc.status = DocumentStatus.completed_with_warnings
                doc.warnings = result.warnings
            else:
                doc.status = DocumentStatus.completed
            log.info("ingestion_succeeded", status=doc.status)
        else:
            doc.status = DocumentStatus.failed
            doc.error_message = result.error
            log.warning("ingestion_failed", error=result.error)

        doc.updated_at = datetime.now(tz=UTC)
        session.commit()
        return result

    except Exception as exc:
        session.rollback()
        try:
            doc = session.get(Document, doc_id)
            if doc:
                doc.status = DocumentStatus.failed
                doc.error_message = str(exc)
                doc.updated_at = datetime.now(tz=UTC)
                session.commit()
        except Exception:
            log.error("failed_to_update_status", error=str(exc))
        raise
    finally:
        session.close()


def create_document_record(
    filename: str,
    file_path: Path,
    file_hash: str,
    file_size_bytes: int,
) -> Document:
    """Create a new document record in the DB and return it."""
    session = get_sync_session()
    try:
        doc = Document(
            filename=filename,
            file_path=str(file_path),
            file_hash=file_hash,
            file_size_bytes=file_size_bytes,
            status=DocumentStatus.pending,
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        logger.info("document_created", doc_id=str(doc.id), filename=filename)
        return doc
    finally:
        session.close()
