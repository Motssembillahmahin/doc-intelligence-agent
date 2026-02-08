"""Unit tests for chunking modules."""

from __future__ import annotations

from pathlib import Path

import fitz

from src.chunking.heading_detector import (
    Heading,
    PageHeadings,
    detect_headings,
    detect_headings_on_page,
    get_current_heading,
)
from src.chunking.splitter import ChunkData, _count_tokens, split_pages
from src.ingestion.pipeline import PageContent
from src.ingestion.table_extractor import ExtractedTable
from src.models.database import ChunkType

# ─── Helpers ──────────────────────────────────────────────────────────────────


def create_pdf_with_headings(tmp_path: Path) -> Path:
    """Create a PDF with headings of different font sizes."""
    doc = fitz.open()
    page = doc.new_page()

    # Large heading (font size 18)
    page.insert_text((72, 72), "Main Title", fontsize=18)
    # Sub-heading (font size 14, bold)
    page.insert_text((72, 120), "Section One", fontsize=14)
    # Body text (font size 11)
    page.insert_text(
        (72, 160),
        "This is regular body text that should not be detected as a heading.",
        fontsize=11,
    )

    pdf_path = tmp_path / "headings.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ─── Heading Detector Tests ──────────────────────────────────────────────────


class TestDetectHeadingsOnPage:
    def test_detects_large_text_as_heading(self, tmp_path: Path):
        pdf_path = create_pdf_with_headings(tmp_path)
        doc = fitz.open(str(pdf_path))
        result = detect_headings_on_page(doc, 0)
        doc.close()

        assert len(result.headings) >= 1
        assert result.page_num == 1
        # At least the 18pt text should be detected
        titles = [h.text for h in result.headings if h.level == 1]
        assert any("Main Title" in t for t in titles)

    def test_body_text_not_detected(self, tmp_path: Path):
        pdf_path = create_pdf_with_headings(tmp_path)
        doc = fitz.open(str(pdf_path))
        result = detect_headings_on_page(doc, 0)
        doc.close()

        heading_texts = [h.text for h in result.headings]
        assert not any("regular body text" in t for t in heading_texts)


class TestDetectHeadings:
    def test_returns_page_headings_for_all_pages(self, tmp_path: Path):
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1} Title", fontsize=16)
        pdf_path = tmp_path / "multi.pdf"
        doc.save(str(pdf_path))
        doc.close()

        results = detect_headings(pdf_path)
        assert len(results) == 3
        for ph in results:
            assert isinstance(ph, PageHeadings)


class TestGetCurrentHeading:
    def test_returns_latest_heading(self):
        all_headings = [
            PageHeadings(
                page_num=1,
                headings=[
                    Heading("Intro", level=1, page_num=1, font_size=18, is_bold=True),
                    Heading("Background", level=2, page_num=1, font_size=14, is_bold=True),
                ],
            ),
            PageHeadings(
                page_num=2,
                headings=[
                    Heading("Methods", level=1, page_num=2, font_size=18, is_bold=True),
                ],
            ),
        ]

        assert get_current_heading(all_headings, 1, "") == "Background"
        assert get_current_heading(all_headings, 2, "") == "Methods"

    def test_returns_none_when_no_headings(self):
        all_headings = [PageHeadings(page_num=1, headings=[])]
        assert get_current_heading(all_headings, 1, "") is None


# ─── Token Counter Tests ─────────────────────────────────────────────────────


class TestCountTokens:
    def test_counts_tokens(self):
        count = _count_tokens("Hello world")
        assert count > 0
        assert isinstance(count, int)

    def test_empty_string(self):
        assert _count_tokens("") == 0

    def test_longer_text_more_tokens(self):
        short = _count_tokens("Hello")
        long = _count_tokens("Hello world, this is a much longer sentence.")
        assert long > short


# ─── Splitter Tests ──────────────────────────────────────────────────────────


class TestSplitPages:
    def test_splits_text_into_chunks(self):
        pages = [
            PageContent(
                page_num=1,
                text="This is a test document. " * 100,
                extraction_method="pymupdf",
            ),
        ]

        chunks = split_pages(pages)
        assert len(chunks) >= 1
        assert all(isinstance(c, ChunkData) for c in chunks)
        assert all(c.chunk_type == ChunkType.text for c in chunks)

    def test_chunk_indices_are_sequential(self):
        pages = [
            PageContent(
                page_num=1,
                text="First page content. " * 50,
                extraction_method="pymupdf",
            ),
            PageContent(
                page_num=2,
                text="Second page content. " * 50,
                extraction_method="pymupdf",
            ),
        ]

        chunks = split_pages(pages)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_tables_become_separate_chunks(self):
        table = ExtractedTable(
            page_num=1,
            table_index=0,
            markdown="| A | B |\n| --- | --- |\n| 1 | 2 |",
            row_count=2,
            col_count=2,
            accuracy=95.0,
            extraction_method="lattice",
        )
        pages = [
            PageContent(
                page_num=1,
                text="Some text content here.",
                extraction_method="pymupdf",
                tables=[table],
            ),
        ]

        chunks = split_pages(pages)
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.table]
        assert len(table_chunks) == 1
        assert "| A | B |" in table_chunks[0].content

    def test_assigns_section_headings(self):
        pages = [
            PageContent(page_num=1, text="Content under intro.", extraction_method="pymupdf"),
            PageContent(page_num=2, text="Content under methods.", extraction_method="pymupdf"),
        ]
        headings = {1: "Introduction", 2: "Methods"}

        chunks = split_pages(pages, headings_by_page=headings)
        page1_chunks = [c for c in chunks if c.page_num == 1]
        page2_chunks = [c for c in chunks if c.page_num == 2]
        assert all(c.section_heading == "Introduction" for c in page1_chunks)
        assert all(c.section_heading == "Methods" for c in page2_chunks)

    def test_token_count_is_positive(self):
        pages = [
            PageContent(page_num=1, text="Hello world.", extraction_method="pymupdf"),
        ]
        chunks = split_pages(pages)
        assert all(c.token_count > 0 for c in chunks)

    def test_empty_page_produces_no_chunks(self):
        pages = [
            PageContent(page_num=1, text="", extraction_method="pymupdf"),
        ]
        chunks = split_pages(pages)
        assert len(chunks) == 0
