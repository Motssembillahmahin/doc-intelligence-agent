"""Query trace persistence and timing utilities."""

from __future__ import annotations

import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager

import structlog
from sqlmodel import Session

from src.models.database import QueryTrace

logger = structlog.get_logger(__name__)


@contextmanager
def timed() -> Generator[dict, None, None]:
    """Context manager that records wall-clock elapsed time in milliseconds.

    Usage::

        with timed() as t:
            do_work()
        elapsed = t["ms"]
    """
    start = time.perf_counter()
    result: dict = {}
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - start) * 1000.0


def save_query_trace(
    *,
    session_id: uuid.UUID,
    original_query: str,
    db_session: Session,
    rewritten_query: str | None = None,
    message_id: uuid.UUID | None = None,
    retrieved_chunks: list | None = None,
    reranked_chunks: list | None = None,
    context_tokens: int | None = None,
    response_tokens: int | None = None,
    total_latency_ms: float | None = None,
    retrieval_latency_ms: float | None = None,
    llm_latency_ms: float | None = None,
) -> QueryTrace | None:
    """Persist a query execution trace to the database.

    Soft-fails on error so that trace recording never breaks the main flow.

    Returns:
        The persisted QueryTrace, or None if saving failed.
    """
    try:
        trace = QueryTrace(
            session_id=session_id,
            message_id=message_id,
            original_query=original_query,
            rewritten_query=rewritten_query,
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            context_tokens=context_tokens,
            response_tokens=response_tokens,
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
        )
        db_session.add(trace)
        db_session.commit()
        logger.debug(
            "query_trace_saved",
            trace_id=str(trace.id),
            session_id=str(session_id),
            total_latency_ms=total_latency_ms,
        )
        return trace
    except Exception as exc:
        logger.warning("query_trace_save_failed", session_id=str(session_id), error=str(exc))
        db_session.rollback()
        return None
