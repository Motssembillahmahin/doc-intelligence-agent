"""Chat service — orchestrates retrieval + LLM for grounded Q&A.

Pipeline:
    1. Load conversation history from DB.
    2. Rewrite query to be self-contained (if history exists).
    3. Retrieve context via hybrid search + reranking.
    4. Check relevance of top-ranked chunks.
    5. Build Anthropic messages with history + embedded context.
    6. Call the LLM.
    7. Persist user and assistant messages.
    8. Compress history if the turn threshold is reached.
"""

from __future__ import annotations

import uuid

import structlog
from sqlmodel import Session as DBSession

from src.llm.client import complete
from src.llm.prompts import SYSTEM_PROMPT, build_messages
from src.llm.relevance import is_context_relevant
from src.memory.history import compress_history, get_history, save_message
from src.memory.query_rewriter import rewrite_query
from src.models.enums import MessageRole
from src.models.schemas import ChatResponse, SourceChunk
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
    log = logger.bind(session_id=str(session_id), query=query[:80])
    log.info("chat_started")

    # Step 1: Load conversation history
    history = get_history(session_id, db_session)

    # Step 2: Rewrite query if there is prior context to resolve
    rewritten = rewrite_query(query, history) if history else query

    # Step 3: Retrieve context using the (possibly rewritten) query
    context = retrieve(rewritten, db_session, doc_ids=doc_ids)

    # Step 4: Relevance gate
    if not is_context_relevant(context.chunks):
        log.info("chat_no_relevant_context", chunks_retrieved=len(context.chunks))
        answer = _NO_CONTEXT_ANSWER
        sources: list[SourceChunk] = []
    else:
        # Step 5 & 6: Build messages and call the LLM
        messages = build_messages(rewritten, context.context_text, history=history)
        response = complete(messages=messages, system=SYSTEM_PROMPT)
        answer = response.content[0].text

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
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    # Step 7: Persist user and assistant messages
    save_message(
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

    # Step 8: Compress old turns if threshold reached
    compress_history(session_id, db_session)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
    )
