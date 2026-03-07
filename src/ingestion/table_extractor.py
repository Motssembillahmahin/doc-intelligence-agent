from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import camelot
import structlog

logger = structlog.get_logger(__name__)


class ExtractionMethod(StrEnum):
    LATTICE = "lattice"
    STREAM = "stream"


@dataclass
class ExtractedTable:
    """A single table extracted from a PDF page."""

    page_num: int  # 1-based
    table_index: int  # Index within the page
    markdown: str
    row_count: int
    col_count: int
    accuracy: float  # Camelot accuracy score
    extraction_method: ExtractionMethod


def _table_to_markdown(table: camelot.core.Table) -> str:
    """Convert a Camelot table to markdown format."""
    df = table.df
    if df.empty:
        return ""

    rows = df.values.tolist()
    if not rows:
        return ""

    header = rows[0]
    md_lines = [
        "| " + " | ".join(str(c).strip() for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(str(c).strip() for c in row) + " |")

    return "\n".join(md_lines)


def extract_tables_from_page(file_path: Path, page_num: int) -> list[ExtractedTable]:
    """Extract tables from a single PDF page.

    Strategy:
    1. Try lattice detection first (ruled/bordered tables)
    2. If no tables found, fall back to stream detection (borderless tables)
    """
    log = logger.bind(file_path=str(file_path), page_num=page_num)
    results: list[ExtractedTable] = []
    page_str = str(page_num)

    try:
        tables = camelot.read_pdf(
            str(file_path),
            pages=page_str,
            flavor=ExtractionMethod.LATTICE,
            suppress_stdout=True,
        )
    except Exception as exc:
        log.debug("lattice_extraction_failed", error=str(exc))
        tables = []

    if tables:
        for idx, table in enumerate(tables):
            md = _table_to_markdown(table)
            if md:
                results.append(
                    ExtractedTable(
                        page_num=page_num,
                        table_index=idx,
                        markdown=md,
                        row_count=table.df.shape[0],
                        col_count=table.df.shape[1],
                        accuracy=table.accuracy,
                        extraction_method=ExtractionMethod.LATTICE,
                    )
                )
        if results:
            log.info("tables_extracted", count=len(results), method=ExtractionMethod.LATTICE)
            return results

    try:
        tables = camelot.read_pdf(
            str(file_path),
            pages=page_str,
            flavor=ExtractionMethod.STREAM,
            suppress_stdout=True,
        )
    except Exception as exc:
        log.debug("stream_extraction_failed", error=str(exc))
        return results

    for idx, table in enumerate(tables):
        md = _table_to_markdown(table)
        if md:
            results.append(
                ExtractedTable(
                    page_num=page_num,
                    table_index=idx,
                    markdown=md,
                    row_count=table.df.shape[0],
                    col_count=table.df.shape[1],
                    accuracy=table.accuracy,
                    extraction_method=ExtractionMethod.STREAM,
                )
            )

    if results:
        log.info("tables_extracted", count=len(results), method="stream")

    return results


def extract_tables(file_path: Path, page_count: int) -> list[ExtractedTable]:
    """Extract tables from all pages of a PDF"""
    log = logger.bind(file_path=str(file_path))
    all_tables: list[ExtractedTable] = []

    for page_num in range(1, page_count + 1):
        try:
            tables = extract_tables_from_page(file_path, page_num)
            all_tables.extend(tables)
        except Exception as exc:
            log.warning("table_extraction_error", page_num=page_num, error=str(exc))

    log.info("table_extraction_complete", total_tables=len(all_tables))
    return all_tables
