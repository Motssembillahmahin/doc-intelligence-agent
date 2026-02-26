"""Relevance checking for retrieved context.

Uses the top cross-encoder rerank score as a proxy for context relevance.
The ms-marco cross-encoder outputs raw logits: positive scores indicate
relevance, negative scores indicate non-relevance. The threshold is
configurable via llm.relevance_threshold in vars.yaml.
"""

from __future__ import annotations

import structlog

from src.config import get_settings
from src.retrieval.reranker import RankedChunk

logger = structlog.get_logger(__name__)


def is_context_relevant(chunks: list[RankedChunk]) -> bool:
    """Determine whether retrieved chunks are relevant enough to answer the query.

    Args:
        chunks: Reranked chunks from the retrieval pipeline, best-first.

    Returns:
        True if at least one chunk exceeds the relevance threshold.
        False if no chunks were retrieved or all scores are below the threshold.
    """
    if not chunks:
        logger.info("relevance_check_failed", reason="no_chunks")
        return False

    settings = get_settings()
    threshold = settings.llm.relevance_threshold
    top_score = chunks[0].rerank_score
    relevant = top_score >= threshold

    logger.info(
        "relevance_check",
        top_score=round(top_score, 4),
        threshold=threshold,
        relevant=relevant,
    )
    return relevant
