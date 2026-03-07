"""Embedding generation and chunk embedding orchestration."""

from __future__ import annotations

import structlog

from src.config import get_settings
from src.embeddings.protocol import VectorStore
from src.models.database import Chunk

logger = structlog.get_logger(__name__)


def _generate_openai(texts: list[str], settings) -> list[list[float]]:
    import openai  # lazy — not imported when provider == "local"

    client = openai.OpenAI(api_key=settings.openai_api_key)
    batch_size = settings.embedding.batch_size
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=settings.embedding.model,
            dimensions=settings.embedding.dimensions,
        )
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


def _generate_local(texts: list[str], settings) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer  # lazy

    model = SentenceTransformer(settings.embedding.local_model)
    return [vec.tolist() for vec in model.encode(texts, convert_to_numpy=True)]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using the configured provider."""
    settings = get_settings()
    provider = settings.embedding.provider
    if provider == "openai":
        return _generate_openai(texts, settings)
    elif provider == "local":
        return _generate_local(texts, settings)
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}. Valid: 'openai', 'local'.")


def embed_chunks(
    chunks: list[Chunk],
    doc_id: str,
    vector_store: VectorStore,
) -> tuple[int, list[str]]:
    """Embed chunks and store in vector store.

    Returns (embedded_count, warnings). Sets chunk.embedding_id in-place.
    """
    if not chunks:
        return 0, []

    log = logger.bind(doc_id=doc_id, total_chunks=len(chunks))
    settings = get_settings()
    batch_size = settings.embedding.batch_size
    embedded_count = 0
    warnings: list[str] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            texts = [c.content for c in batch]
            embeddings = generate_embeddings(texts)

            ids = [str(c.id) for c in batch]
            metadatas = [
                {
                    "doc_id": doc_id,
                    "page_num": c.page_num,
                    "chunk_type": c.chunk_type.value
                    if hasattr(c.chunk_type, "value")
                    else str(c.chunk_type),
                    "section_heading": c.section_heading or "",
                }
                for c in batch
            ]

            vector_store.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts,
            )

            for c in batch:
                c.embedding_id = str(c.id)

            embedded_count += len(batch)

        except Exception as exc:
            batch_start = i
            batch_end = min(i + batch_size, len(chunks))
            msg = f"Embedding failed for chunks {batch_start}-{batch_end - 1}: {exc}"
            warnings.append(msg)
            log.warning("embedding_batch_failed", batch_start=batch_start, error=str(exc))

    log.info("embedding_complete", embedded=embedded_count, warnings=len(warnings))
    return embedded_count, warnings
