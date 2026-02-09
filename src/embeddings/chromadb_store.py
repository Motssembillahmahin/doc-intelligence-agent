"""ChromaDB implementation of the VectorStore protocol."""

from __future__ import annotations

from typing import Any

import chromadb
import structlog

from src.config import get_settings
from src.embeddings.protocol import VectorSearchResult

logger = structlog.get_logger(__name__)


class ChromaDBStore:
    """VectorStore backed by ChromaDB over HTTP."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chromadb.collection_name,
            metadata={"hnsw:space": settings.chromadb.distance_function},
        )
        logger.info(
            "chromadb_connected",
            collection=settings.chromadb.collection_name,
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def query(
        self,
        embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results: list[VectorSearchResult] = []
        if raw["ids"] and raw["ids"][0]:
            ids = raw["ids"][0]
            distances = raw["distances"][0] if raw["distances"] else [0.0] * len(ids)
            metadatas = raw["metadatas"][0] if raw["metadatas"] else [{}] * len(ids)
            for chunk_id, dist, meta in zip(ids, distances, metadatas, strict=True):
                results.append(
                    VectorSearchResult(
                        chunk_id=chunk_id,
                        score=1.0 - dist,
                        metadata=meta,
                    )
                )
        return results

    def delete_by_doc_id(self, doc_id: str) -> int:
        existing = self._collection.get(where={"doc_id": doc_id})
        ids_to_delete = existing["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
