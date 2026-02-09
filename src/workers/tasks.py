"""Celery task definitions — thin wrappers around service functions."""

from __future__ import annotations

import uuid
from pathlib import Path

import structlog

from src.services.ingestion_service import ingest_document
from src.workers.celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(name="ingest_document", bind=True, max_retries=1)
def ingest_document_task(self, doc_id: str, file_path: str) -> dict:
    """Background task to ingest a PDF document."""
    log = logger.bind(doc_id=doc_id, task_id=self.request.id)
    log.info("task_started")

    try:
        result = ingest_document(uuid.UUID(doc_id), Path(file_path))
        return {
            "doc_id": doc_id,
            "success": result.success,
            "page_count": result.validation.page_count,
            "warnings": result.warnings,
            "error": result.error,
        }
    except Exception as exc:
        log.error("task_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=30) from exc
