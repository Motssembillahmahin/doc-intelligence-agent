from __future__ import annotations

import enum
import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlmodel import Field, SQLModel


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    SKIPPED = "skipped"


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    filename: str
    file_hash: str = Field(unique=True, index=True)
    status: DocumentStatus = Field(
        sa_column=sa.Column(
            sa.Enum(DocumentStatus, name="document_status", native_enum=True),
            nullable=False,
            default=DocumentStatus.PENDING,
        )
    )
    total_pages: int | None = None
    file_size_bytes: int | None = None
    error_message: str | None = None
    warnings: list | None = Field(default=None, sa_column=sa.Column(sa.JSON))
    metadata_: dict | None = Field(
        default=None, sa_column=sa.Column("metadata", sa.JSON)
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=sa.Column(sa.DateTime, server_default=sa.func.now()),
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=sa.Column(
            sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()
        ),
    )


class Session(SQLModel, table=True):
    __tablename__ = "sessions"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    title: str | None = None
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=sa.Column(sa.DateTime, server_default=sa.func.now()),
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=sa.Column(
            sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()
        ),
    )


class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="sessions.id", index=True)
    role: str
    content: str
    rewritten_query: str | None = None
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=sa.Column(sa.DateTime, server_default=sa.func.now()),
    )


class QueryTrace(SQLModel, table=True):
    __tablename__ = "query_traces"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="sessions.id", index=True)
    original_query: str
    rewritten_query: str | None = None
    retrieved_chunks: list | None = Field(default=None, sa_column=sa.Column(sa.JSON))
    scores: list | None = Field(default=None, sa_column=sa.Column(sa.JSON))
    response: str | None = None
    latency_ms: float | None = None
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=sa.Column(sa.DateTime, server_default=sa.func.now()),
    )


class Metric(SQLModel, table=True):
    __tablename__ = "metrics"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(index=True)
    value: float
    labels: dict | None = Field(default=None, sa_column=sa.Column(sa.JSON))
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=sa.Column(sa.DateTime, server_default=sa.func.now()),
    )
