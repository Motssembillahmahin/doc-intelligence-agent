"""PDF text extraction — page-by-page using PyMuPDF with pdfplumber fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import structlog

logger = structlog.get_logger(__name__)

# Minimum characters on a page to consider it "has text"
MIN_TEXT_LENGTH = 20


@dataclass
class PageText:
    """Extracted text for a single PDF page."""

    page_num: int  # 1-based
    text: str
    extraction_method: str  # "pymupdf" or "pdfplumber"
    has_text: bool = True
    warnings: list[str] = field(default_factory=list)


def extract_page_text_pymupdf(doc: fitz.Document, page_idx: int) -> str:
    """Extract text from a single page using PyMuPDF."""
    page = doc[page_idx]
    return page.get_text("text")


def extract_page_text_pdfplumber(file_path: Path, page_idx: int) -> str:
    """Extract text from a single page using pdfplumber (fallback)."""
    with pdfplumber.open(str(file_path)) as pdf:
        page = pdf.pages[page_idx]
        text = page.extract_text() or ""
    return text


def extract_text(file_path: Path) -> list[PageText]:
    """Extract text from all pages of a PDF, page by page.

    Strategy:
    1. Try PyMuPDF first (fast, handles most PDFs well)
    2. If PyMuPDF returns very little text for a page, fall back to pdfplumber
    3. If both return little text, mark the page as needing OCR

    Returns a list of PageText objects, one per page.
    """
    log = logger.bind(file_path=str(file_path))
    results: list[PageText] = []

    doc = fitz.open(str(file_path))
    page_count = len(doc)
    log.info("starting_text_extraction", page_count=page_count)

    for page_idx in range(page_count):
        page_num = page_idx + 1  # 1-based
        page_log = log.bind(page_num=page_num)

        # Try PyMuPDF first
        text = extract_page_text_pymupdf(doc, page_idx)

        if len(text.strip()) >= MIN_TEXT_LENGTH:
            results.append(
                PageText(
                    page_num=page_num,
                    text=text.strip(),
                    extraction_method="pymupdf",
                )
            )
            continue

        # Fall back to pdfplumber
        page_log.debug("pymupdf_low_text, trying_pdfplumber")
        try:
            text_fallback = extract_page_text_pdfplumber(file_path, page_idx)
        except Exception as exc:
            page_log.warning("pdfplumber_failed", error=str(exc))
            text_fallback = ""

        if len(text_fallback.strip()) >= MIN_TEXT_LENGTH:
            results.append(
                PageText(
                    page_num=page_num,
                    text=text_fallback.strip(),
                    extraction_method="pdfplumber",
                )
            )
            continue

        # Both extractors returned very little — mark as needing OCR
        fallback_stripped = text_fallback.strip()
        text_stripped = text.strip()
        best_text = (
            fallback_stripped if len(fallback_stripped) > len(text_stripped) else text_stripped
        )
        results.append(
            PageText(
                page_num=page_num,
                text=best_text,
                extraction_method="pymupdf",
                has_text=False,
                warnings=[f"Low text content ({len(best_text)} chars), may need OCR"],
            )
        )
        page_log.info("page_low_text", char_count=len(best_text))

    doc.close()
    log.info(
        "text_extraction_complete",
        total_pages=page_count,
        pages_with_text=sum(1 for p in results if p.has_text),
        pages_needing_ocr=sum(1 for p in results if not p.has_text),
    )
    return results
