"""PDF validation — file size, page count, format, and corruption checks."""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

import fitz  # PyMuPDF
import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)


class ValidationError(Exception):
    """Raised when a PDF fails validation."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ValidationResult:
    """Result of PDF validation."""

    def __init__(
        self,
        *,
        valid: bool,
        file_path: Path,
        file_hash: str = "",
        file_size_bytes: int = 0,
        page_count: int = 0,
        error: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.valid = valid
        self.file_path = file_path
        self.file_hash = file_hash
        self.file_size_bytes = file_size_bytes
        self.page_count = page_count
        self.error = error
        self.warnings = warnings or []


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_pdf(file_path: Path) -> ValidationResult:
    """Validate a PDF file for ingestion.

    Checks:
    1. File exists and is readable
    2. File extension is supported
    3. MIME type matches PDF
    4. File size within configured limit
    5. PDF is not corrupted (can be opened by PyMuPDF)
    6. Page count > 0
    """
    settings = get_settings()
    log = logger.bind(file_path=str(file_path))
    warnings: list[str] = []

    if not file_path.exists():
        log.warning("file_not_found")
        return ValidationResult(valid=False, file_path=file_path, error="File not found")

    if not file_path.is_file():
        log.warning("not_a_file")
        return ValidationResult(valid=False, file_path=file_path, error="Path is not a file")

    suffix = file_path.suffix.lower()
    if suffix not in settings.ingestion.supported_formats:
        log.warning("unsupported_format", suffix=suffix)
        return ValidationResult(
            valid=False,
            file_path=file_path,
            error=(
                f"Unsupported format: {suffix}. Supported: {settings.ingestion.supported_formats}"
            ),
        )

    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type and mime_type != "application/pdf":
        log.warning("mime_mismatch", mime_type=mime_type)
        warnings.append(f"MIME type mismatch: {mime_type}")

    file_size_bytes = file_path.stat().st_size
    max_size_bytes = settings.ingestion.max_file_size_mb * 1024 * 1024
    if file_size_bytes > max_size_bytes:
        log.warning("file_too_large", size_mb=file_size_bytes / (1024 * 1024))
        return ValidationResult(
            valid=False,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            error=(
                f"File size {file_size_bytes / (1024 * 1024):.1f}MB"
                f" exceeds limit of {settings.ingestion.max_file_size_mb}MB"
            ),
        )

    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        log.warning("pdf_corrupt", error=str(exc))
        return ValidationResult(
            valid=False,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            error=f"PDF appears corrupt: {exc}",
        )

    page_count = len(doc)
    doc.close()

    if page_count == 0:
        log.warning("zero_pages")
        return ValidationResult(
            valid=False,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            error="PDF has 0 pages",
        )

    file_hash = compute_file_hash(file_path)

    log.info("validation_passed", page_count=page_count, file_size_bytes=file_size_bytes)

    return ValidationResult(
        valid=True,
        file_path=file_path,
        file_hash=file_hash,
        file_size_bytes=file_size_bytes,
        page_count=page_count,
        warnings=warnings,
    )
