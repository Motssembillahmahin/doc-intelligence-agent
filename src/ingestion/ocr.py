from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import structlog
from PIL import Image

from src.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class OCRResult:
    """OCR result for a single page."""

    page_num: int  # 1-based
    text: str
    success: bool
    error: str | None = None


def ocr_page(doc: fitz.Document, page_idx: int) -> OCRResult:
    """Run OCR on a single PDF page.

    Renders the page as a high-DPI image, then runs pytesseract.
    """
    import pytesseract

    settings = get_settings()
    page_num = page_idx + 1
    log = logger.bind(page_num=page_num)

    try:
        page = doc[page_idx]
        # Render page at configured DPI
        dpi = settings.ingestion.image_dpi
        zoom = dpi / 72  # PDF default is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Run OCR
        text = pytesseract.image_to_string(img, lang=settings.ingestion.ocr_language)

        log.info("ocr_complete", char_count=len(text.strip()))
        return OCRResult(page_num=page_num, text=text.strip(), success=True)

    except Exception as exc:
        log.warning("ocr_failed", error=str(exc))
        return OCRResult(page_num=page_num, text="", success=False, error=str(exc))


def ocr_pages(file_path: Path, page_nums: list[int]) -> list[OCRResult]:
    """Run OCR on specific pages of a PDF"""
    settings = get_settings()
    log = logger.bind(file_path=str(file_path))

    if not settings.ingestion.ocr_enabled:
        log.info("ocr_disabled")
        return []

    if not page_nums:
        return []

    log.info("starting_ocr", page_count=len(page_nums))
    results: list[OCRResult] = []

    doc = fitz.open(str(file_path))
    for page_num in page_nums:
        page_idx = page_num - 1  # Convert to 0-based
        result = ocr_page(doc, page_idx)
        results.append(result)

    doc.close()

    successful = sum(1 for r in results if r.success)
    log.info("ocr_complete", total=len(results), successful=successful)
    return results
