"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlmodel import Session

from src.api.routes import chat, documents, sessions, summaries
from src.db.engine import get_sync_engine
from src.models.schemas import HealthResponse

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("api_startup")
    yield
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Document Intelligence API",
        version="0.1.0",
        description="RAG-based PDF ingestion and conversational Q&A platform.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(summaries.router)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health_check() -> HealthResponse:
        """Check liveness of the database and ChromaDB."""
        db_status = "ok"
        try:
            with Session(get_sync_engine()) as session:
                session.execute(text("SELECT 1"))
        except Exception as exc:
            db_status = f"error: {exc}"

        chroma_status = "ok"
        try:
            from src.embeddings.chromadb_store import ChromaDBStore

            ChromaDBStore()
        except Exception as exc:
            chroma_status = f"error: {exc}"

        overall = "ok" if db_status == "ok" and chroma_status == "ok" else "degraded"
        return HealthResponse(status=overall, database=db_status, chromadb=chroma_status)

    return app


app = create_app()
