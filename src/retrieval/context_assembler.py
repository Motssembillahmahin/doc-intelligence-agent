"""Context assembly from reranked chunks, respecting a token budget.

Each chunk is formatted with a numbered source header:
    [N] | Page P | § Section Heading
    <chunk content>

Chunks are added in rerank order until the token budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
import tiktoken

from src.config import get_settings
from src.retrieval.reranker import RankedChunk

logger = structlog.get_logger(__name__)


@dataclass
class AssembledContext:
    chunks: list[RankedChunk]
    context_text: str
    total_tokens: int
    truncated: bool = False


def _format_chunk(chunk: RankedChunk, index: int) -> str:
    """Format a single chunk with its source metadata header."""
    header_parts = [f"[{index}]", f"Page {chunk.page_num}"]
    if chunk.section_heading:
        header_parts.append(f"§ {chunk.section_heading}")
    header = " | ".join(header_parts)
    return f"{header}\n{chunk.content}"


def assemble_context(
    chunks: list[RankedChunk],
    max_context_tokens: int | None = None,
) -> AssembledContext:
    """Assemble reranked chunks into a context string within a token budget.

    Args:
        chunks: Reranked chunks in best-first order.
        max_context_tokens: Hard token cap for the assembled context text.
                            Defaults to retrieval.max_context_tokens from settings.

    Returns:
        AssembledContext containing the selected chunks, formatted context
        string, total token count, and a flag indicating truncation.
    """
    if not chunks:
        return AssembledContext(chunks=[], context_text="", total_tokens=0)

    settings = get_settings()
    if max_context_tokens is not None:
        token_limit = max_context_tokens
    else:
        token_limit = settings.retrieval.max_context_tokens
    enc = tiktoken.get_encoding(settings.chunking.tiktoken_encoding)

    selected: list[RankedChunk] = []
    total_tokens = 0
    truncated = False

    for chunk in chunks:
        formatted = _format_chunk(chunk, len(selected) + 1)
        tokens = len(enc.encode(formatted))

        if total_tokens + tokens > token_limit:
            truncated = True
            logger.debug(
                "context_budget_exceeded",
                chunk_index=len(selected),
                tokens_so_far=total_tokens,
                limit=token_limit,
            )
            break

        selected.append(chunk)
        total_tokens += tokens

    context_text = "\n\n---\n\n".join(_format_chunk(c, i + 1) for i, c in enumerate(selected))

    logger.info(
        "context_assembled",
        chunks_used=len(selected),
        total_tokens=total_tokens,
        truncated=truncated,
    )

    return AssembledContext(
        chunks=selected,
        context_text=context_text,
        total_tokens=total_tokens,
        truncated=truncated,
    )
