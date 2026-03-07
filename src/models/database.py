"""Database models for the Document Intelligence System."""

import uuid
from datetime import UTC, datetime
from typing import Optional

from sqlalchemy import Column, Index
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlmodel import Field, Relationship, SQLModel

from src.models.enums import ChunkType, DocumentStatus, MessageRole


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    filename: str = Field(index=True)
    file_path: str
    file_hash: str = Field(index=True)  # SHA-256 for change detection
    file_size_bytes: int
    page_count: int | None = None
    status: DocumentStatus = Field(default=DocumentStatus.pending, index=True)
    warnings: list | None = Field(default=None, sa_column=Column(JSONB))
    error_message: str | None = None
    summary: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    # Relationships
    chunks: list["Chunk"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "passive_deletes": True},
    )


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"
    __table_args__ = (Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    doc_id: uuid.UUID = Field(foreign_key="documents.id", index=True)
    content: str
    chunk_type: ChunkType = Field(default=ChunkType.text)
    page_num: int
    section_heading: str | None = None
    chunk_index: int  # Position within the document
    token_count: int
    chunk_metadata: dict | None = Field(default=None, sa_column=Column(JSONB))
    embedding_id: str | None = None  # Reference to ChromaDB vector ID
    search_vector: str | None = Field(default=None, sa_column=Column(TSVECTOR))
    created_at: datetime = Field(default_factory=_utc_now)

    # Relationships
    document: Document | None = Relationship(back_populates="chunks")


class Session(SQLModel, table=True):
    __tablename__ = "sessions"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    title: str | None = None
    summary: str | None = None  # Compressed conversation history
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    # Relationships
    messages: list["Message"] = Relationship(
        back_populates="session",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "passive_deletes": True},
    )
    query_traces: list["QueryTrace"] = Relationship(
        back_populates="session",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "passive_deletes": True},
    )


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="sessions.id", index=True)
    role: MessageRole
    content: str
    original_query: str | None = None  # Pre-rewrite user query
    rewritten_query: str | None = None  # Post-rewrite for context resolution
    created_at: datetime = Field(default_factory=_utc_now)

    # Relationships
    session: Session | None = Relationship(back_populates="messages")
    query_trace: Optional["QueryTrace"] = Relationship(back_populates="message")


class QueryTrace(SQLModel, table=True):
    __tablename__ = "query_traces"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    message_id: uuid.UUID | None = Field(default=None, foreign_key="messages.id", unique=True)
    session_id: uuid.UUID = Field(foreign_key="sessions.id", index=True)
    original_query: str
    rewritten_query: str | None = None
    retrieved_chunks: list | None = Field(default=None, sa_column=Column(JSONB))
    reranked_chunks: list | None = Field(default=None, sa_column=Column(JSONB))
    context_tokens: int | None = None
    response_tokens: int | None = None
    total_latency_ms: float | None = None
    retrieval_latency_ms: float | None = None
    llm_latency_ms: float | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    # Relationships
    session: Session | None = Relationship(back_populates="query_traces")
    message: Message | None = Relationship(back_populates="query_trace")


class Metric(SQLModel, table=True):
    __tablename__ = "metrics"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    metric_name: str = Field(index=True)
    metric_value: float
    labels: dict | None = Field(default=None, sa_column=Column(JSONB))
    created_at: datetime = Field(default_factory=_utc_now)
