from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel

from src.models.enums import DocumentStatus


class DocumentUploadResponse(BaseModel):
    id: uuid.UUID
    filename: str
    status: DocumentStatus


class DocumentStatusResponse(BaseModel):
    id: uuid.UUID
    filename: str
    status: DocumentStatus
    total_pages: int | None = None
    file_size_bytes: int | None = None
    error_message: str | None = None
    warnings: list | None = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentStatusResponse]
    total: int


class SourceChunk(BaseModel):
    doc_id: uuid.UUID
    page_num: int
    chunk_type: str
    section_heading: str | None = None
    content: str
    score: float


class ChatRequest(BaseModel):
    session_id: uuid.UUID
    query: str


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    answer: str
    sources: list[SourceChunk]


class SessionResponse(BaseModel):
    id: uuid.UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    database: str
    chromadb: str
