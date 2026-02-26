"""Hybrid search combining ChromaDB vector search and PostgreSQL FTS.

Results from both sources are fused using alpha-weighted normalized scores:
    fusion_score = alpha * norm_vector_score + (1 - alpha) * norm_fts_score
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import func, select, text
from sqlmodel import Session

from src.config import get_settings
from src.embeddings.protocol import VectorStore

logger = structlog.get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    page_num: int
    chunk_type: str
    section_heading: str | None
    content: str
    vector_score: float | None
    fts_score: float | None
    fusion_score: float = 0.0


def _vector_search(
    query_embedding: list[float],
    vector_store: VectorStore,
    n_results: int,
    doc_ids: list[str] | None,
) -> list[RetrievedChunk]:
    """Query ChromaDB for nearest neighbours."""
    where = None
    if doc_ids:
        where = {"doc_id": {"$in": doc_ids}} if len(doc_ids) > 1 else {"doc_id": doc_ids[0]}

    results = vector_store.query(
        embedding=query_embedding,
        n_results=n_results,
        where=where,
    )

    chunks: list[RetrievedChunk] = []
    for r in results:
        meta = r.metadata
        chunks.append(
            RetrievedChunk(
                chunk_id=r.chunk_id,
                doc_id=meta.get("doc_id", ""),
                page_num=int(meta.get("page_num", 0)),
                chunk_type=meta.get("chunk_type", "text"),
                section_heading=meta.get("section_heading") or None,
                content=r.document or "",
                vector_score=r.score,
                fts_score=None,
            )
        )
    return chunks


def _fts_search(
    query_text: str,
    session: Session,
    n_results: int,
    doc_ids: list[str] | None,
) -> list[RetrievedChunk]:
    """Full-text search using PostgreSQL tsvector."""
    from src.models.database import Chunk

    tsquery = func.plainto_tsquery("english", query_text)
    rank_col = func.ts_rank_cd(Chunk.search_vector, tsquery).label("rank")

    stmt = (
        select(
            Chunk.id,
            Chunk.doc_id,
            Chunk.page_num,
            Chunk.chunk_type,
            Chunk.section_heading,
            Chunk.content,
            rank_col,
        )
        .where(Chunk.search_vector.op("@@")(tsquery))
        .order_by(text("rank DESC"))
        .limit(n_results)
    )

    if doc_ids:
        stmt = stmt.where(Chunk.doc_id.in_([uuid.UUID(d) for d in doc_ids]))

    rows = session.execute(stmt).fetchall()

    chunks: list[RetrievedChunk] = []
    for row in rows:
        chunks.append(
            RetrievedChunk(
                chunk_id=str(row.id),
                doc_id=str(row.doc_id),
                page_num=row.page_num,
                chunk_type=row.chunk_type,
                section_heading=row.section_heading,
                content=row.content,
                vector_score=None,
                fts_score=float(row.rank),
            )
        )
    return chunks


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _fuse(
    vector_chunks: list[RetrievedChunk],
    fts_chunks: list[RetrievedChunk],
    alpha: float,
) -> list[RetrievedChunk]:
    """Fuse vector and FTS results using alpha-weighted normalized scores.

    FTS chunks are authoritative for content (fetched directly from DB).
    Vector-only chunks use the document text stored in ChromaDB.
    """
    # Index FTS chunks first — they carry DB content
    merged: dict[str, RetrievedChunk] = {}
    for c in fts_chunks:
        merged[c.chunk_id] = c

    # Merge vector results: add vector_score; add new entries for vector-only hits
    for c in vector_chunks:
        if c.chunk_id in merged:
            merged[c.chunk_id].vector_score = c.vector_score
        else:
            merged[c.chunk_id] = c

    chunks = list(merged.values())

    # Normalize each score distribution independently before combining
    vec_scores = [c.vector_score if c.vector_score is not None else 0.0 for c in chunks]
    fts_scores = [c.fts_score if c.fts_score is not None else 0.0 for c in chunks]

    norm_vec = _normalize(vec_scores)
    norm_fts = _normalize(fts_scores)

    for chunk, nv, nf in zip(chunks, norm_vec, norm_fts, strict=True):
        chunk.fusion_score = alpha * nv + (1.0 - alpha) * nf

    chunks.sort(key=lambda c: c.fusion_score, reverse=True)
    return chunks


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    vector_store: VectorStore,
    session: Session,
    doc_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Run hybrid search and return fused, ranked results.

    Args:
        query_text: Raw query string for FTS.
        query_embedding: Pre-computed query embedding for vector search.
        vector_store: VectorStore implementation (e.g. ChromaDBStore).
        session: Synchronous SQLModel/SQLAlchemy session.
        doc_ids: Optional list of document UUID strings to restrict the search.

    Returns:
        Fused and sorted list of RetrievedChunk, best-first.
    """
    settings = get_settings()
    top_k = settings.retrieval.top_k
    alpha = settings.retrieval.alpha

    log = logger.bind(query=query_text[:80], top_k=top_k, alpha=alpha)

    vector_chunks = _vector_search(query_embedding, vector_store, top_k, doc_ids)
    fts_chunks = _fts_search(query_text, session, top_k, doc_ids)

    log.info(
        "hybrid_search_raw",
        vector_results=len(vector_chunks),
        fts_results=len(fts_chunks),
    )

    if not vector_chunks and not fts_chunks:
        return []

    fused = _fuse(vector_chunks, fts_chunks, alpha)
    log.info("hybrid_search_fused", total_results=len(fused))
    return fused
