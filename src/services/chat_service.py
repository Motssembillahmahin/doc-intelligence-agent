"""Chat service — orchestrates retrieval + LLM for grounded Q&A.

Pipeline:
    1. Retrieve context via hybrid search + reranking.
    2. Check relevance of top-ranked chunks.
    3. Build Anthropic messages with embedded context.
    4. Call the LLM.
    5. Return ChatResponse with answer and source citations.
"""

from __future__ import annotations

import uuid

import structlog
from sqlmodel import Session

from src.llm.client import complete
from src.llm.prompts import SYSTEM_PROMPT, build_messages
from src.llm.relevance import is_context_relevant
from src.models.schemas import ChatResponse, SourceChunk
from src.services.retrieval_service import retrieve

logger = structlog.get_logger(__name__)

_NO_CONTEXT_ANSWER = (
    "I don't have enough information in the provided documents to answer this question."
)


def chat(
    query: str,
    session_id: uuid.UUID,
    db_session: Session,
    doc_ids: list[str] | None = None,
    history: list[dict] | None = None,
) -> ChatResponse:
    """Answer a user query using retrieval-augmented generation.

    Args:
        query: The user's question.
        session_id: UUID of the conversation session (included in the response).
        db_session: Synchronous DB session (caller owns lifecycle).
        doc_ids: Optional list of document UUID strings to restrict retrieval scope.
        history: Prior conversation turns as [{"role": ..., "content": ...}].
                 Only raw Q&A text — context is not re-embedded from history.

    Returns:
        ChatResponse containing the answer and a list of cited source chunks.
    """
    log = logger.bind(session_id=str(session_id), query=query[:80])
    log.info("chat_started")

    # Step 1: Retrieve context
    context = retrieve(query, db_session, doc_ids=doc_ids)

    # Step 2: Relevance gate — short-circuit if context is insufficient
    if not is_context_relevant(context.chunks):
        log.info("chat_no_relevant_context", chunks_retrieved=len(context.chunks))
        return ChatResponse(
            session_id=session_id,
            answer=_NO_CONTEXT_ANSWER,
            sources=[],
        )

    # Step 3: Build messages and call the LLM
    messages = build_messages(query, context.context_text, history=history)
    response = complete(messages=messages, system=SYSTEM_PROMPT)
    answer = response.content[0].text

    # Step 4: Map context chunks to the SourceChunk response schema
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

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
    )
