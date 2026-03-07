"""Cross-encoder reranking of retrieved chunks.

The cross-encoder model is loaded once per process (lazy singleton) to avoid
repeated cold-start overhead in Celery workers and FastAPI.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from sentence_transformers import CrossEncoder

from src.config import get_settings
from src.retrieval.hybrid_search import RetrievedChunk

logger = structlog.get_logger(__name__)

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        settings = get_settings()
        _reranker = CrossEncoder(settings.retrieval.rerank_model)
        logger.info("reranker_loaded", model=settings.retrieval.rerank_model)
    return _reranker


@dataclass
class RankedChunk:
    chunk_id: str
    doc_id: str
    page_num: int
    chunk_type: str
    section_heading: str | None
    content: str
    fusion_score: float
    rerank_score: float


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RankedChunk]:
    """Re-rank chunks using a cross-encoder model.

    Args:
        query: The user query string.
        chunks: Candidate chunks from hybrid search.
        top_k: How many chunks to return after reranking. Defaults to
               retrieval.rerank_top_k from settings.

    Returns:
        List of RankedChunk sorted by rerank_score descending, truncated to top_k.
    """
    if not chunks:
        return []

    settings = get_settings()
    k = top_k if top_k is not None else settings.retrieval.rerank_top_k

    reranker = _get_reranker()
    pairs = [(query, c.content) for c in chunks]
    scores: list[float] = reranker.predict(pairs).tolist()

    ranked: list[RankedChunk] = [
        RankedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            page_num=chunk.page_num,
            chunk_type=chunk.chunk_type,
            section_heading=chunk.section_heading,
            content=chunk.content,
            fusion_score=chunk.fusion_score,
            rerank_score=score,
        )
        for chunk, score in zip(chunks, scores, strict=True)
    ]

    ranked.sort(key=lambda r: r.rerank_score, reverse=True)
    result = ranked[:k]

    logger.info(
        "reranking_complete",
        input_chunks=len(chunks),
        output_chunks=len(result),
        top_score=result[0].rerank_score if result else None,
    )
    return result
