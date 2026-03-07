"""Text splitting for document chunks using LangChain + tiktoken."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.ingestion.pipeline import PageContent
from src.models.enums import ChunkType

logger = structlog.get_logger(__name__)


@dataclass
class ChunkData:
    content: str
    chunk_type: ChunkType
    page_num: int
    chunk_index: int
    section_heading: str | None
    token_count: int
    metadata: dict = field(default_factory=dict)


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken with encoding from config."""
    settings = get_settings()
    enc = tiktoken.get_encoding(settings.chunking.tiktoken_encoding)
    return len(enc.encode(text))


def _create_splitter() -> RecursiveCharacterTextSplitter:
    """Create a text splitter from config settings."""
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=_count_tokens,
    )


def split_pages(
    pages: list[PageContent],
    headings_by_page: dict[int, str | None] | None = None,
) -> list[ChunkData]:
    """Split page contents into chunks"""
    log = logger.bind(page_count=len(pages))
    splitter = _create_splitter()
    chunks: list[ChunkData] = []
    chunk_index = 0

    for page in pages:
        current_heading = headings_by_page.get(page.page_num) if headings_by_page else None

        if page.text:
            text_chunks = splitter.split_text(page.text)
            for text in text_chunks:
                chunks.append(
                    ChunkData(
                        content=text,
                        chunk_type=ChunkType.text,
                        page_num=page.page_num,
                        chunk_index=chunk_index,
                        section_heading=current_heading,
                        token_count=_count_tokens(text),
                    )
                )
                chunk_index += 1

        for table in page.tables:
            chunks.append(
                ChunkData(
                    content=table.markdown,
                    chunk_type=ChunkType.table,
                    page_num=page.page_num,
                    chunk_index=chunk_index,
                    section_heading=current_heading,
                    token_count=_count_tokens(table.markdown),
                    metadata={"accuracy": table.accuracy, "method": table.extraction_method},
                )
            )
            chunk_index += 1

    log.info("splitting_complete", total_chunks=len(chunks))
    return chunks
