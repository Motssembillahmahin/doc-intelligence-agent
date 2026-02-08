"""Ingestion pipeline orchestrator — coordinates validation, extraction, OCR, and images."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog

from src.ingestion.image_extractor import ExtractedImage, extract_images
from src.ingestion.ocr import ocr_pages
from src.ingestion.table_extractor import ExtractedTable, extract_tables
from src.ingestion.text_extractor import extract_text
from src.ingestion.validator import ValidationResult, validate_pdf

logger = structlog.get_logger(__name__)


@dataclass
class PageContent:
    """All extracted content for a single page."""

    page_num: int
    text: str
    extraction_method: str  # "pymupdf", "pdfplumber", or "ocr"
    tables: list[ExtractedTable] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)


@dataclass
class IngestionResult:
    """Complete result of the ingestion pipeline."""

    validation: ValidationResult
    pages: list[PageContent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.validation.valid and self.error is None

    @property
    def total_tables(self) -> int:
        return sum(len(p.tables) for p in self.pages)

    @property
    def total_images(self) -> int:
        return sum(len(p.images) for p in self.pages)


def run_ingestion(file_path: Path, doc_id: str) -> IngestionResult:
    """Run the full ingestion pipeline on a PDF file.

    Steps:
    1. Validate the PDF
    2. Extract text page-by-page (PyMuPDF → pdfplumber fallback)
    3. OCR pages with low/no text
    4. Extract tables from all pages
    5. Extract images from all pages
    """
    log = logger.bind(doc_id=doc_id, file_path=str(file_path))
    all_warnings: list[str] = []

    log.info("pipeline_step", step="validation")
    validation = validate_pdf(file_path)

    if not validation.valid:
        log.warning("validation_failed", error=validation.error)
        return IngestionResult(validation=validation, error=validation.error)

    all_warnings.extend(validation.warnings)
    page_count = validation.page_count

    log.info("pipeline_step", step="text_extraction")
    try:
        page_texts = extract_text(file_path)
    except Exception as exc:
        log.error("text_extraction_failed", error=str(exc))
        return IngestionResult(
            validation=validation,
            error=f"Text extraction failed: {exc}",
            warnings=all_warnings,
        )

    page_map: dict[int, PageContent] = {}
    ocr_needed: list[int] = []

    for pt in page_texts:
        page_map[pt.page_num] = PageContent(
            page_num=pt.page_num,
            text=pt.text,
            extraction_method=pt.extraction_method,
        )
        all_warnings.extend(pt.warnings)
        if not pt.has_text:
            ocr_needed.append(pt.page_num)

    if ocr_needed:
        log.info("pipeline_step", step="ocr", pages=ocr_needed)
        try:
            ocr_results = ocr_pages(file_path, ocr_needed)
            for ocr_result in ocr_results:
                if ocr_result.success and ocr_result.text:
                    page = page_map.get(ocr_result.page_num)
                    if page:
                        page.text = ocr_result.text
                        page.extraction_method = "ocr"
                elif not ocr_result.success:
                    all_warnings.append(
                        f"OCR failed on page {ocr_result.page_num}: {ocr_result.error}"
                    )
        except Exception as exc:
            log.warning("ocr_step_failed", error=str(exc))
            all_warnings.append(f"OCR step failed: {exc}")

    log.info("pipeline_step", step="table_extraction")
    try:
        tables = extract_tables(file_path, page_count)
        for table in tables:
            page = page_map.get(table.page_num)
            if page:
                page.tables.append(table)
    except Exception as exc:
        log.warning("table_extraction_failed", error=str(exc))
        all_warnings.append(f"Table extraction failed: {exc}")

    log.info("pipeline_step", step="image_extraction")
    try:
        images = extract_images(file_path, doc_id, page_count)
        for image in images:
            page = page_map.get(image.page_num)
            if page:
                page.images.append(image)
    except Exception as exc:
        log.warning("image_extraction_failed", error=str(exc))
        all_warnings.append(f"Image extraction failed: {exc}")

    pages = [page_map[pn] for pn in sorted(page_map.keys())]

    result = IngestionResult(
        validation=validation,
        pages=pages,
        warnings=all_warnings,
    )

    log.info(
        "pipeline_complete",
        page_count=len(pages),
        total_tables=result.total_tables,
        total_images=result.total_images,
        warnings_count=len(all_warnings),
    )

    return result
