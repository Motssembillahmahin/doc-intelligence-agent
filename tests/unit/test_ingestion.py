"""Unit tests for the ingestion pipeline modules."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF

from src.ingestion.image_extractor import extract_images
from src.ingestion.ocr import ocr_page, ocr_pages
from src.ingestion.pipeline import PageContent, run_ingestion
from src.ingestion.table_extractor import _table_to_markdown
from src.ingestion.text_extractor import extract_text
from src.ingestion.validator import (
    compute_file_hash,
    validate_pdf,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────


def create_test_pdf(
    tmp_path: Path,
    *,
    pages: int = 3,
    text: str = "This is a sample page with enough text for testing the extraction pipeline.",
    filename: str = "test.pdf",
) -> Path:
    """Create a simple test PDF with text content."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}\n\n{text}")
    pdf_path = tmp_path / filename
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def create_empty_pdf(tmp_path: Path) -> Path:
    """Create a PDF with 0 pages (which PyMuPDF can't actually do, so we create a corrupt file)."""
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 corrupt content")
    return pdf_path


def create_large_pdf(tmp_path: Path, size_mb: int = 60) -> Path:
    """Create a PDF that exceeds the size limit by writing padding."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Large file test")
    pdf_path = tmp_path / "large.pdf"
    doc.save(str(pdf_path))
    doc.close()
    # Pad the file to exceed limit
    with open(pdf_path, "ab") as f:
        f.write(b"\x00" * (size_mb * 1024 * 1024))
    return pdf_path


# ─── Validator Tests ──────────────────────────────────────────────────────────


class TestComputeFileHash:
    def test_returns_sha256(self, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        result = compute_file_hash(test_file)
        expected = hashlib.sha256(b"hello").hexdigest()
        assert result == expected

    def test_consistent_hash(self, tmp_path: Path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("consistent content")
        assert compute_file_hash(test_file) == compute_file_hash(test_file)


class TestValidatePdf:
    def test_valid_pdf(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path)
        result = validate_pdf(pdf_path)
        assert result.valid is True
        assert result.page_count == 3
        assert result.file_size_bytes > 0
        assert result.file_hash != ""
        assert result.error is None

    def test_file_not_found(self, tmp_path: Path):
        result = validate_pdf(tmp_path / "nonexistent.pdf")
        assert result.valid is False
        assert "not found" in result.error.lower()

    def test_not_a_file(self, tmp_path: Path):
        result = validate_pdf(tmp_path)
        assert result.valid is False
        assert "not a file" in result.error.lower()

    def test_unsupported_format(self, tmp_path: Path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        result = validate_pdf(txt_file)
        assert result.valid is False
        assert "unsupported format" in result.error.lower()

    def test_file_too_large(self, tmp_path: Path):
        pdf_path = create_large_pdf(tmp_path, size_mb=60)
        result = validate_pdf(pdf_path)
        assert result.valid is False
        assert "exceeds limit" in result.error.lower()

    def test_corrupt_pdf(self, tmp_path: Path):
        pdf_path = tmp_path / "corrupt.pdf"
        pdf_path.write_bytes(b"this is not a valid pdf")
        result = validate_pdf(pdf_path)
        assert result.valid is False
        assert "corrupt" in result.error.lower()

    def test_single_page_pdf(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path, pages=1)
        result = validate_pdf(pdf_path)
        assert result.valid is True
        assert result.page_count == 1


# ─── Text Extractor Tests ────────────────────────────────────────────────────


class TestExtractText:
    def test_extracts_text_from_all_pages(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path, pages=3)
        results = extract_text(pdf_path)
        assert len(results) == 3
        for i, page_text in enumerate(results):
            assert page_text.page_num == i + 1
            assert "sample page" in page_text.text.lower()
            assert page_text.has_text is True

    def test_page_numbers_are_1_based(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path, pages=2)
        results = extract_text(pdf_path)
        assert results[0].page_num == 1
        assert results[1].page_num == 2

    def test_extraction_method_is_pymupdf(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path, pages=1)
        results = extract_text(pdf_path)
        assert results[0].extraction_method == "pymupdf"

    def test_low_text_page_marked_for_ocr(self, tmp_path: Path):
        """Page with very little text should be flagged."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "x")  # Very short text
        pdf_path = tmp_path / "low_text.pdf"
        doc.save(str(pdf_path))
        doc.close()

        results = extract_text(pdf_path)
        assert len(results) == 1
        assert results[0].has_text is False
        assert len(results[0].warnings) > 0


# ─── Table Extractor Tests ───────────────────────────────────────────────────


class TestTableToMarkdown:
    def test_converts_table_to_markdown(self):
        mock_table = MagicMock()
        import pandas as pd

        mock_table.df = pd.DataFrame(
            {
                "Name": ["Alice", "Bob"],
                "Age": ["30", "25"],
            }
        )
        # The first row of df.values becomes the header
        md = _table_to_markdown(mock_table)
        assert "| Name |" in md or "| Alice |" in md
        assert "---" in md

    def test_empty_table_returns_empty_string(self):
        mock_table = MagicMock()
        import pandas as pd

        mock_table.df = pd.DataFrame()
        md = _table_to_markdown(mock_table)
        assert md == ""


# ─── OCR Tests ────────────────────────────────────────────────────────────────


class TestOCR:
    @patch.dict("sys.modules", {"pytesseract": MagicMock()})
    def test_ocr_page_success(self, tmp_path: Path):
        import sys

        mock_tesseract = sys.modules["pytesseract"]
        mock_tesseract.image_to_string.return_value = "OCR extracted text content here"

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "barely visible")

        result = ocr_page(doc, 0)
        doc.close()

        assert result.success is True
        assert result.page_num == 1
        assert result.text == "OCR extracted text content here"

    def test_ocr_pages_with_ocr_disabled(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path, pages=1)
        with patch("src.ingestion.ocr.get_settings") as mock_settings:
            mock_settings.return_value.ingestion.ocr_enabled = False
            results = ocr_pages(pdf_path, [1])
        assert results == []

    def test_ocr_pages_with_empty_list(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path, pages=1)
        results = ocr_pages(pdf_path, [])
        assert results == []


# ─── Image Extractor Tests ───────────────────────────────────────────────────


class TestImageExtractor:
    def test_extract_images_from_text_only_pdf(self, tmp_path: Path):
        """A text-only PDF should yield no images."""
        pdf_path = create_test_pdf(tmp_path, pages=2)
        output_dir = tmp_path / "images"
        results = extract_images(pdf_path, "test-doc-id", 2, output_dir=output_dir)
        assert results == []

    def test_extract_images_creates_output_dir(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path, pages=1)
        output_dir = tmp_path / "new_dir" / "images"
        extract_images(pdf_path, "test-doc-id", 1, output_dir=output_dir)
        assert output_dir.exists()


# ─── Pipeline Tests ──────────────────────────────────────────────────────────


class TestIngestionPipeline:
    def test_successful_ingestion(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path, pages=2)
        result = run_ingestion(pdf_path, "test-doc-id")
        assert result.success is True
        assert len(result.pages) == 2
        assert result.validation.valid is True
        assert result.error is None

    def test_invalid_file_returns_error(self, tmp_path: Path):
        result = run_ingestion(tmp_path / "nonexistent.pdf", "test-doc-id")
        assert result.success is False
        assert result.error is not None

    def test_pages_contain_text(self, tmp_path: Path):
        pdf_path = create_test_pdf(
            tmp_path, pages=1, text="Hello world from the test PDF document."
        )
        result = run_ingestion(pdf_path, "test-doc-id")
        assert result.success is True
        assert len(result.pages) == 1
        assert "hello world" in result.pages[0].text.lower()

    def test_ingestion_result_properties(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path, pages=1)
        result = run_ingestion(pdf_path, "test-doc-id")
        assert result.total_tables >= 0
        assert result.total_images >= 0

    def test_corrupt_pdf_fails_gracefully(self, tmp_path: Path):
        pdf_path = tmp_path / "corrupt.pdf"
        pdf_path.write_bytes(b"not a pdf file at all")
        result = run_ingestion(pdf_path, "test-doc-id")
        assert result.success is False

    def test_page_content_structure(self, tmp_path: Path):
        pdf_path = create_test_pdf(tmp_path, pages=2)
        result = run_ingestion(pdf_path, "test-doc-id")
        for page in result.pages:
            assert isinstance(page, PageContent)
            assert page.page_num > 0
            assert isinstance(page.text, str)
            assert isinstance(page.tables, list)
            assert isinstance(page.images, list)
