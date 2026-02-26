"""Retrieval service — orchestrates the full retrieval pipeline.

Pipeline:
    1. Embed the query via OpenAI embeddings.
    2. Run hybrid search (vector + PostgreSQL FTS) with score fusion.
    3. Rerank candidates with a cross-encoder.
    4. Assemble the final context string within a token budget.
"""

from __future__ import annotations

import structlog
from sqlmodel import Session

from src.config import get_settings
from src.embeddings.chromadb_store import ChromaDBStore
from src.embeddings.embedder import generate_embeddings
from src.retrieval.context_assembler import AssembledContext, assemble_context
from src.retrieval.hybrid_search import RetrievedChunk, hybrid_search
from src.retrieval.reranker import RankedChunk, rerank

logger = structlog.get_logger(__name__)


def retrieve(
    query: str,
    session: Session,
    doc_ids: list[str] | None = None,
    max_context_tokens: int | None = None,
) -> AssembledContext:
    """Run the full retrieval pipeline for a user query.

    Args:
        query: The user's question or search string.
        session: A synchronous SQLModel/SQLAlchemy session (caller owns lifecycle).
        doc_ids: Optional list of document UUID strings to restrict the search
                 scope. Pass None to search across all documents.
        max_context_tokens: Override the token cap for context assembly.
                            Defaults to retrieval.max_context_tokens in settings.

    Returns:
        AssembledContext with ranked chunks and a formatted context string
        ready for use in an LLM prompt.
    """
    log = logger.bind(query=query[:80], doc_ids=doc_ids)
    settings = get_settings()

    log.info("retrieval_started")

    # Step 1: Embed the query
    query_embeddings = generate_embeddings([query])
    query_embedding = query_embeddings[0]

    # Step 2: Hybrid search — vector + FTS, fused by alpha
    vector_store = ChromaDBStore()
    raw_chunks: list[RetrievedChunk] = hybrid_search(
        query_text=query,
        query_embedding=query_embedding,
        vector_store=vector_store,
        session=session,
        doc_ids=doc_ids,
    )

    if not raw_chunks:
        log.info("retrieval_no_results")
        return AssembledContext(chunks=[], context_text="", total_tokens=0)

    # Step 3: Rerank candidates with cross-encoder
    ranked_chunks: list[RankedChunk] = rerank(
        query=query,
        chunks=raw_chunks,
        top_k=settings.retrieval.rerank_top_k,
    )

    # Step 4: Assemble context within token budget
    context = assemble_context(ranked_chunks, max_context_tokens=max_context_tokens)

    log.info(
        "retrieval_complete",
        raw_chunks=len(raw_chunks),
        ranked_chunks=len(ranked_chunks),
        context_chunks=len(context.chunks),
        context_tokens=context.total_tokens,
        truncated=context.truncated,
    )
    return context
