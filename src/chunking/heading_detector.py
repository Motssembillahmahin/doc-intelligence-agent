"""Heading detection from PDF font metadata using PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import structlog

logger = structlog.get_logger(__name__)

# Font size thresholds for heading classification
HEADING_MIN_FONT_SIZE = 13.0
SUBHEADING_MIN_FONT_SIZE = 11.0


@dataclass
class Heading:
    """A detected heading with its position and level."""

    text: str
    level: int  # 1 = top-level, 2 = sub-heading, etc.
    page_num: int  # 1-based
    font_size: float
    is_bold: bool


@dataclass
class PageHeadings:
    """All headings detected on a single page."""

    page_num: int
    headings: list[Heading] = field(default_factory=list)


def _is_bold(font_name: str, flags: int) -> bool:
    """Check if a font span is bold based on name or flags."""
    # flags bit 4 (16) = bold in PyMuPDF
    if flags & (1 << 4):
        return True
    font_lower = font_name.lower()
    return "bold" in font_lower or "black" in font_lower


def detect_headings_on_page(doc: fitz.Document, page_idx: int) -> PageHeadings:
    """Detect headings on a single PDF page using font metadata.

    Uses font size and boldness to classify text spans as headings.
    Larger/bold text that forms short lines is likely a heading.
    """
    page_num = page_idx + 1
    page = doc[page_idx]
    blocks = page.get_text("dict")["blocks"]

    headings: list[Heading] = []

    for block in blocks:
        if block.get("type") != 0:  # 0 = text block
            continue

        for line in block.get("lines", []):
            line_text_parts: list[str] = []
            max_font_size = 0.0
            has_bold = False

            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                line_text_parts.append(text)
                font_size = span.get("size", 0.0)
                max_font_size = max(max_font_size, font_size)
                if _is_bold(span.get("font", ""), span.get("flags", 0)):
                    has_bold = True

            line_text = " ".join(line_text_parts).strip()
            if not line_text or len(line_text) > 200:
                continue

            # Classify heading level
            level = 0
            if max_font_size >= HEADING_MIN_FONT_SIZE:
                level = 1
            elif max_font_size >= SUBHEADING_MIN_FONT_SIZE and has_bold:
                level = 2
            elif has_bold and len(line_text) < 80:
                level = 3

            if level > 0:
                headings.append(
                    Heading(
                        text=line_text,
                        level=level,
                        page_num=page_num,
                        font_size=max_font_size,
                        is_bold=has_bold,
                    )
                )

    return PageHeadings(page_num=page_num, headings=headings)


def detect_headings(file_path: Path) -> list[PageHeadings]:
    """Detect headings across all pages of a PDF.

    Returns a list of PageHeadings, one per page.
    """
    log = logger.bind(file_path=str(file_path))
    doc = fitz.open(str(file_path))
    results: list[PageHeadings] = []

    for page_idx in range(len(doc)):
        page_headings = detect_headings_on_page(doc, page_idx)
        results.append(page_headings)

    doc.close()

    total = sum(len(ph.headings) for ph in results)
    log.info("heading_detection_complete", total_headings=total)
    return results


def get_current_heading(all_headings: list[PageHeadings], page_num: int) -> str | None:
    """Get the most recent heading at or before a given page.

    Useful for assigning section_heading metadata to chunks.
    Walks backwards through pages to find the latest heading.
    """
    latest_heading: str | None = None
    for ph in all_headings:
        if ph.page_num > page_num:
            break
        for heading in ph.headings:
            latest_heading = heading.text
    return latest_heading
