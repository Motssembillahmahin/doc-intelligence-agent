"""Ingestion service — orchestrates PDF ingestion, chunking, and DB persistence."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import structlog

from src.chunking.heading_detector import detect_headings, get_current_heading
from src.chunking.splitter import split_pages
from src.db.engine import get_sync_session
from src.ingestion.pipeline import IngestionResult, run_ingestion
from src.models.database import Chunk, Document
from src.models.enums import DocumentStatus

logger = structlog.get_logger(__name__)


def _build_headings_map(file_path: Path) -> dict[int, str | None]:
    """Detect headings and build a page_num → current_heading map."""
    all_headings = detect_headings(file_path)
    headings_map: dict[int, str | None] = {}
    for ph in all_headings:
        headings_map[ph.page_num] = get_current_heading(all_headings, ph.page_num)
    return headings_map


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

            # Chunking: detect headings, split pages, persist chunks
            log.info("chunking_started")
            headings_by_page = _build_headings_map(file_path)
            chunk_data_list = split_pages(result.pages, headings_by_page=headings_by_page)

            for cd in chunk_data_list:
                chunk = Chunk(
                    doc_id=doc_id,
                    content=cd.content,
                    chunk_type=cd.chunk_type,
                    page_num=cd.page_num,
                    chunk_index=cd.chunk_index,
                    section_heading=cd.section_heading,
                    token_count=cd.token_count,
                    chunk_metadata=cd.metadata if cd.metadata else None,
                )
                session.add(chunk)

            log.info("chunking_complete", total_chunks=len(chunk_data_list))

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
