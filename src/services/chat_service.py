"""Chat service — orchestrates retrieval + LLM for grounded Q&A.

Pipeline:
    1. Load conversation history from DB.
    2. Rewrite query to be self-contained (if history exists).
    3. [timed] Retrieve context via hybrid search + reranking.
    4. Check relevance of top-ranked chunks.
    5. [timed] Build Anthropic messages and call the LLM.
    6. Persist user message (before response, to obtain message_id).
    7. Persist assistant message.
    8. Save QueryTrace (if observability.enable_traces).
    9. Record latency and token metrics.
    10. Compress history if the turn threshold is reached.
"""

from __future__ import annotations

import uuid

import structlog
from sqlmodel import Session as DBSession

from src.config import get_settings
from src.llm.client import complete
from src.llm.prompts import SYSTEM_PROMPT, build_messages
from src.llm.relevance import is_context_relevant
from src.memory.history import compress_history, get_history, save_message
from src.memory.query_rewriter import rewrite_query
from src.models.enums import MessageRole
from src.models.schemas import ChatResponse, SourceChunk
from src.observability.metrics import record_metric
from src.observability.traces import save_query_trace, timed
from src.services.retrieval_service import retrieve

logger = structlog.get_logger(__name__)

_NO_CONTEXT_ANSWER = (
    "I don't have enough information in the provided documents to answer this question."
)


def chat(
    query: str,
    session_id: uuid.UUID,
    db_session: DBSession,
    doc_ids: list[str] | None = None,
) -> ChatResponse:
    """Answer a user query using retrieval-augmented generation.

    Args:
        query: The raw user question.
        session_id: UUID of the conversation session.
        db_session: Synchronous DB session (caller owns lifecycle).
        doc_ids: Optional list of document UUID strings to restrict retrieval scope.

    Returns:
        ChatResponse containing the answer and cited source chunks.
    """
    settings = get_settings()
    log = logger.bind(session_id=str(session_id), query=query[:80])
    log.info("chat_started")

    input_tokens: int | None = None
    output_tokens: int | None = None
    retrieval_ms: float | None = None
    llm_ms: float | None = None

    # Step 1: Load conversation history
    history = get_history(session_id, db_session)

    # Step 2: Rewrite query if there is prior context to resolve
    rewritten = rewrite_query(query, history) if history else query

    # Step 3: Retrieve context (timed)
    with timed() as retrieval_timer:
        result = retrieve(rewritten, db_session, doc_ids=doc_ids)
    retrieval_ms = retrieval_timer["ms"]
    context = result.context

    # Step 4: Relevance gate
    if not is_context_relevant(context.chunks):
        log.info("chat_no_relevant_context", chunks_retrieved=len(context.chunks))
        answer = _NO_CONTEXT_ANSWER
        sources: list[SourceChunk] = []
    else:
        # Step 5: Build messages and call LLM (timed)
        messages = build_messages(rewritten, context.context_text, history=history)
        with timed() as llm_timer:
            response = complete(messages=messages, system=SYSTEM_PROMPT)
        llm_ms = llm_timer["ms"]
        answer = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        sources = [
            SourceChunk(
                doc_id=uuid.UUID(chunk.doc_id),
                page_num=chunk.page_num,
                chunk_type=chunk.chunk_type,
                section_heading=chunk.section_heading,
                content=chunk.content,
                score=chunk.rerank_score,
            )
            for chunk in context.chunks
        ]

        log.info(
            "chat_complete",
            answer_length=len(answer),
            sources=len(sources),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retrieval_ms=round(retrieval_ms, 1),
            llm_ms=round(llm_ms, 1),
        )

    total_ms = retrieval_ms + (llm_ms or 0.0)

    # Step 6 & 7: Persist messages — user first to obtain message_id for trace
    user_msg = save_message(
        session_id=session_id,
        role=MessageRole.user,
        content=query,
        db_session=db_session,
        original_query=query,
        rewritten_query=rewritten if rewritten != query else None,
    )
    save_message(
        session_id=session_id,
        role=MessageRole.assistant,
        content=answer,
        db_session=db_session,
    )

    # Step 8: Persist query trace
    if settings.observability.enable_traces:
        save_query_trace(
            session_id=session_id,
            message_id=user_msg.id,
            original_query=query,
            rewritten_query=rewritten if rewritten != query else None,
            retrieved_chunks=[{"chunk_id": cid} for cid in result.raw_chunk_ids],
            reranked_chunks=[{"chunk_id": cid} for cid in result.reranked_chunk_ids],
            context_tokens=context.total_tokens,
            response_tokens=output_tokens,
            total_latency_ms=total_ms,
            retrieval_latency_ms=retrieval_ms,
            llm_latency_ms=llm_ms,
            db_session=db_session,
        )

    # Step 9: Record metrics
    session_label = {"session_id": str(session_id)}
    record_metric("chat.latency_ms", total_ms, db_session, labels=session_label)
    if input_tokens is not None:
        record_metric("chat.input_tokens", input_tokens, db_session, labels=session_label)
    if output_tokens is not None:
        record_metric("chat.output_tokens", output_tokens, db_session, labels=session_label)

    # Step 10: Compress old turns if threshold reached
    compress_history(session_id, db_session)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
    )
