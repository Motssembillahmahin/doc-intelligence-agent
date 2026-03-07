"""Map-reduce summarization for a single document's chunks.

Map step:  each chunk  → short summary (LLM call per chunk)
Reduce step: batch summaries hierarchically until one final summary remains.
"""

from __future__ import annotations

import structlog

from src.config import get_settings
from src.llm.client import complete
from src.models.database import Chunk
from src.models.enums import ChunkType
from src.summarization.prompts import MAP_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


def _map_chunk(content: str, max_tokens: int) -> str:
    """Summarize a single chunk (map step)."""
    response = complete(
        messages=[{"role": "user", "content": content}],
        system=MAP_SYSTEM_PROMPT,
        max_tokens=max_tokens,
    )
    return response.content[0].text.strip()


def _reduce_batch(summaries: list[str], max_tokens: int) -> str:
    """Combine a batch of partial summaries into one (reduce step)."""
    combined = "\n\n".join(f"[Section {i + 1}]\n{s}" for i, s in enumerate(summaries))
    response = complete(
        messages=[{"role": "user", "content": combined}],
        system=REDUCE_SYSTEM_PROMPT,
        max_tokens=max_tokens,
    )
    return response.content[0].text.strip()


def _hierarchical_reduce(summaries: list[str], batch_size: int, max_tokens: int) -> str:
    """Reduce a list of summaries hierarchically until a single summary remains."""
    while len(summaries) > 1:
        batches = [summaries[i : i + batch_size] for i in range(0, len(summaries), batch_size)]
        summaries = [_reduce_batch(batch, max_tokens) for batch in batches]
    return summaries[0]


def summarize_document_chunks(chunks: list[Chunk], doc_id: str) -> str:
    """Run map-reduce summarization over a document's chunks.

    Image caption chunks are skipped as they rarely carry standalone summary value.

    Args:
        chunks: DB Chunk objects for the document, ordered by chunk_index.
        doc_id: Document UUID string (for logging).

    Returns:
        Final document summary string, or "" if no usable chunks.
    """
    settings = get_settings()
    s = settings.summarization
    log = logger.bind(doc_id=doc_id, total_chunks=len(chunks))

    # Prefer text and table chunks; fall back to all chunks if nothing else
    text_chunks = [c for c in chunks if c.chunk_type != ChunkType.image_caption]
    if not text_chunks:
        text_chunks = chunks

    if not text_chunks:
        return ""

    log.info("map_started", chunks=len(text_chunks))

    # Map step: summarize each chunk individually
    chunk_summaries: list[str] = []
    for i, chunk in enumerate(text_chunks):
        try:
            summary = _map_chunk(chunk.content, max_tokens=s.map_max_tokens)
            chunk_summaries.append(summary)
        except Exception as exc:
            log.warning("map_chunk_failed", chunk_index=i, error=str(exc))

    if not chunk_summaries:
        return ""

    log.info("reduce_started", partial_summaries=len(chunk_summaries))

    # Reduce step: hierarchically combine until one summary remains
    final = _hierarchical_reduce(
        chunk_summaries,
        batch_size=s.max_chunks_per_reduce_batch,
        max_tokens=s.reduce_max_tokens,
    )

    log.info("summarization_complete", summary_length=len(final))
    return final
