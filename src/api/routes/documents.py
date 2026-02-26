"""Document routes — upload, list, status, delete."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlmodel import Session, select

from src.api.deps import get_db
from src.config import get_settings
from src.config.settings import PROJECT_ROOT
from src.embeddings.chromadb_store import ChromaDBStore
from src.models.database import Document
from src.models.enums import DocumentStatus
from src.models.schemas import DocumentListResponse, DocumentStatusResponse, DocumentUploadResponse
from src.services.ingestion_service import create_document_record
from src.workers.tasks import ingest_document_task

router = APIRouter(prefix="/documents", tags=["documents"])

logger = structlog.get_logger(__name__)


def _doc_to_status(doc: Document) -> DocumentStatusResponse:
    return DocumentStatusResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        total_pages=doc.page_count,
        file_size_bytes=doc.file_size_bytes,
        error_message=doc.error_message,
        warnings=doc.warnings,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def upload_document(
    file: Annotated[UploadFile, File()],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentUploadResponse:
    """Upload a PDF document and dispatch background ingestion."""
    settings = get_settings()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ingestion.supported_formats:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported format '{suffix}'. Accepted: {settings.ingestion.supported_formats}",  # noqa: E501
        )

    data = file.file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.ingestion.max_file_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File {size_mb:.1f} MB exceeds limit of {settings.ingestion.max_file_size_mb} MB",  # noqa: E501
        )

    file_hash = hashlib.sha256(data).hexdigest()

    # Dedup: return existing document if already ingested successfully
    existing = db.exec(select(Document).where(Document.file_hash == file_hash)).first()
    if existing and existing.status in (
        DocumentStatus.completed,
        DocumentStatus.completed_with_warnings,
    ):
        logger.info("document_deduplicated", doc_id=str(existing.id), hash=file_hash[:12])
        return DocumentUploadResponse(
            id=existing.id,
            filename=existing.filename,
            status=DocumentStatus.skipped,
        )

    # Persist file to disk
    upload_dir = PROJECT_ROOT / settings.ingestion.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename or 'upload').name}"
    file_path = upload_dir / safe_name
    file_path.write_bytes(data)

    # Create DB record (owns its own session) then dispatch Celery task
    doc = create_document_record(
        filename=file.filename or safe_name,
        file_path=file_path,
        file_hash=file_hash,
        file_size_bytes=len(data),
    )
    ingest_document_task.delay(str(doc.id), str(file_path))

    logger.info("document_uploaded", doc_id=str(doc.id), filename=doc.filename)
    return DocumentUploadResponse(id=doc.id, filename=doc.filename, status=doc.status)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    db: Annotated[Session, Depends(get_db)],
    limit: int = 50,
    offset: int = 0,
) -> DocumentListResponse:
    """List all documents ordered by upload time."""
    docs = db.exec(
        select(Document).order_by(Document.created_at.desc()).offset(offset).limit(limit)
    ).all()
    total = db.exec(select(func.count()).select_from(Document)).one()
    return DocumentListResponse(
        documents=[_doc_to_status(d) for d in docs],
        total=total,
    )


@router.get("/{doc_id}", response_model=DocumentStatusResponse)
def get_document(
    doc_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> DocumentStatusResponse:
    """Get status and metadata for a specific document."""
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return _doc_to_status(doc)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    doc_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete a document, its chunks, vector embeddings, and file from disk."""
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    # Best-effort: delete from vector store
    try:
        ChromaDBStore().delete_by_doc_id(str(doc_id))
    except Exception as exc:
        logger.warning("vector_delete_failed", doc_id=str(doc_id), error=str(exc))

    # Best-effort: remove file from disk
    try:
        Path(doc.file_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("file_delete_failed", doc_id=str(doc_id), error=str(exc))

    db.delete(doc)
    db.commit()
