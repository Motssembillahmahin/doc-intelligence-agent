"""Vector store protocol — abstraction layer for embedding storage backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class VectorSearchResult:
    chunk_id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None: ...

    def query(
        self,
        embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]: ...

    def delete_by_doc_id(self, doc_id: str) -> int: ...
