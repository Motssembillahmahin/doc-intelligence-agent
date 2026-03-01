"""Unit tests for the embeddings module."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.embeddings.chromadb_store import ChromaDBStore
from src.embeddings.embedder import embed_chunks, generate_embeddings
from src.embeddings.protocol import VectorSearchResult


class TestVectorSearchResult:
    def test_creation(self):
        result = VectorSearchResult(chunk_id="abc", score=0.95, metadata={"doc_id": "123"})
        assert result.chunk_id == "abc"
        assert result.score == 0.95
        assert result.metadata == {"doc_id": "123"}

    def test_fields(self):
        result = VectorSearchResult(chunk_id="x", score=0.0, metadata={})
        assert result.chunk_id == "x"
        assert result.score == 0.0
        assert result.metadata == {}


class TestChromaDBStore:
    @patch("src.embeddings.chromadb_store.get_settings")
    @patch("src.embeddings.chromadb_store.chromadb.HttpClient")
    def test_init_creates_collection(self, mock_client_cls, mock_settings):
        settings = MagicMock()
        settings.chroma_host = "localhost"
        settings.chroma_port = 8100
        settings.chromadb.collection_name = "test_col"
        settings.chromadb.distance_function = "cosine"
        mock_settings.return_value = settings

        client = MagicMock()
        mock_client_cls.return_value = client

        store = ChromaDBStore()

        mock_client_cls.assert_called_once_with(host="localhost", port=8100)
        client.get_or_create_collection.assert_called_once_with(
            name="test_col",
            metadata={"hnsw:space": "cosine"},
        )
        assert store._collection == client.get_or_create_collection.return_value

    @patch("src.embeddings.chromadb_store.get_settings")
    @patch("src.embeddings.chromadb_store.chromadb.HttpClient")
    def test_add_calls_upsert(self, mock_client_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            chroma_host="h",
            chroma_port=1,
            chromadb=MagicMock(collection_name="c", distance_function="cosine"),
        )
        mock_client_cls.return_value = MagicMock()

        store = ChromaDBStore()
        store.add(
            ids=["1", "2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"k": "v"}, {"k": "v2"}],
            documents=["doc1", "doc2"],
        )

        store._collection.upsert.assert_called_once_with(
            ids=["1", "2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"k": "v"}, {"k": "v2"}],
            documents=["doc1", "doc2"],
        )

    @patch("src.embeddings.chromadb_store.get_settings")
    @patch("src.embeddings.chromadb_store.chromadb.HttpClient")
    def test_query_returns_results(self, mock_client_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            chroma_host="h",
            chroma_port=1,
            chromadb=MagicMock(collection_name="c", distance_function="cosine"),
        )
        mock_client_cls.return_value = MagicMock()

        store = ChromaDBStore()
        store._collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"doc_id": "d1"}, {"doc_id": "d2"}]],
        }

        results = store.query([0.1, 0.2], n_results=5)

        assert len(results) == 2
        assert results[0].chunk_id == "id1"
        assert results[0].score == pytest.approx(0.9)
        assert results[1].chunk_id == "id2"
        assert results[1].score == pytest.approx(0.7)

    @patch("src.embeddings.chromadb_store.get_settings")
    @patch("src.embeddings.chromadb_store.chromadb.HttpClient")
    def test_query_with_filter(self, mock_client_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            chroma_host="h",
            chroma_port=1,
            chromadb=MagicMock(collection_name="c", distance_function="cosine"),
        )
        mock_client_cls.return_value = MagicMock()

        store = ChromaDBStore()
        store._collection.query.return_value = {
            "ids": [["id1"]],
            "distances": [[0.05]],
            "metadatas": [[{"doc_id": "d1"}]],
        }

        results = store.query([0.1], n_results=3, where={"doc_id": "d1"})

        store._collection.query.assert_called_once_with(
            query_embeddings=[[0.1]],
            n_results=3,
            where={"doc_id": "d1"},
        )
        assert len(results) == 1

    @patch("src.embeddings.chromadb_store.get_settings")
    @patch("src.embeddings.chromadb_store.chromadb.HttpClient")
    def test_query_empty_results(self, mock_client_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            chroma_host="h",
            chroma_port=1,
            chromadb=MagicMock(collection_name="c", distance_function="cosine"),
        )
        mock_client_cls.return_value = MagicMock()

        store = ChromaDBStore()
        store._collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        results = store.query([0.1], n_results=5)
        assert results == []

    @patch("src.embeddings.chromadb_store.get_settings")
    @patch("src.embeddings.chromadb_store.chromadb.HttpClient")
    def test_delete_by_doc_id(self, mock_client_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            chroma_host="h",
            chroma_port=1,
            chromadb=MagicMock(collection_name="c", distance_function="cosine"),
        )
        mock_client_cls.return_value = MagicMock()

        store = ChromaDBStore()
        store._collection.get.return_value = {"ids": ["id1", "id2", "id3"]}

        count = store.delete_by_doc_id("doc-123")

        store._collection.get.assert_called_once_with(where={"doc_id": "doc-123"})
        store._collection.delete.assert_called_once_with(ids=["id1", "id2", "id3"])
        assert count == 3

    @patch("src.embeddings.chromadb_store.get_settings")
    @patch("src.embeddings.chromadb_store.chromadb.HttpClient")
    def test_delete_by_doc_id_noop(self, mock_client_cls, mock_settings):
        mock_settings.return_value = MagicMock(
            chroma_host="h",
            chroma_port=1,
            chromadb=MagicMock(collection_name="c", distance_function="cosine"),
        )
        mock_client_cls.return_value = MagicMock()

        store = ChromaDBStore()
        store._collection.get.return_value = {"ids": []}

        count = store.delete_by_doc_id("doc-nonexistent")

        store._collection.delete.assert_not_called()
        assert count == 0


class TestGenerateEmbeddings:
    @patch("src.embeddings.embedder.get_settings")
    @patch("openai.OpenAI")
    def test_single_batch(self, mock_openai_cls, mock_settings):
        settings = MagicMock()
        settings.openai_api_key = "test-key"
        settings.embedding.provider = "openai"
        settings.embedding.batch_size = 100
        settings.embedding.model = "text-embedding-3-small"
        settings.embedding.dimensions = 1536
        mock_settings.return_value = settings

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_item1 = MagicMock()
        mock_item1.embedding = [0.1, 0.2]
        mock_item2 = MagicMock()
        mock_item2.embedding = [0.3, 0.4]
        mock_client.embeddings.create.return_value = MagicMock(data=[mock_item1, mock_item2])

        result = generate_embeddings(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embeddings.create.assert_called_once_with(
            input=["hello", "world"],
            model="text-embedding-3-small",
            dimensions=1536,
        )

    @patch("src.embeddings.embedder.get_settings")
    @patch("openai.OpenAI")
    def test_multiple_batches(self, mock_openai_cls, mock_settings):
        settings = MagicMock()
        settings.openai_api_key = "test-key"
        settings.embedding.provider = "openai"
        settings.embedding.batch_size = 2
        settings.embedding.model = "text-embedding-3-small"
        settings.embedding.dimensions = 1536
        mock_settings.return_value = settings

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        batch1_items = [MagicMock(embedding=[0.1]), MagicMock(embedding=[0.2])]
        batch2_items = [MagicMock(embedding=[0.3])]
        mock_client.embeddings.create.side_effect = [
            MagicMock(data=batch1_items),
            MagicMock(data=batch2_items),
        ]

        result = generate_embeddings(["a", "b", "c"])

        assert result == [[0.1], [0.2], [0.3]]
        assert mock_client.embeddings.create.call_count == 2

    @patch("src.embeddings.embedder.get_settings")
    @patch("openai.OpenAI")
    def test_order_preservation(self, mock_openai_cls, mock_settings):
        settings = MagicMock()
        settings.openai_api_key = "test-key"
        settings.embedding.provider = "openai"
        settings.embedding.batch_size = 1
        settings.embedding.model = "m"
        settings.embedding.dimensions = 3
        mock_settings.return_value = settings

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.embeddings.create.side_effect = [
            MagicMock(data=[MagicMock(embedding=[1.0])]),
            MagicMock(data=[MagicMock(embedding=[2.0])]),
            MagicMock(data=[MagicMock(embedding=[3.0])]),
        ]

        result = generate_embeddings(["x", "y", "z"])
        assert result == [[1.0], [2.0], [3.0]]


class TestGenerateEmbeddingsLocal:
    @patch("src.embeddings.embedder.get_settings")
    @patch("src.embeddings.embedder._generate_local")
    def test_dispatches_to_local(self, mock_local, mock_settings):
        settings = MagicMock()
        settings.embedding.provider = "local"
        mock_settings.return_value = settings
        mock_local.return_value = [[0.1, 0.2]]
        result = generate_embeddings(["hello"])
        assert result == [[0.1, 0.2]]
        mock_local.assert_called_once_with(["hello"], settings)

    @patch("src.embeddings.embedder.get_settings")
    @patch("sentence_transformers.SentenceTransformer")
    def test_local_encode_called(self, mock_st_cls, mock_settings):
        import numpy as np

        settings = MagicMock()
        settings.embedding.provider = "local"
        settings.embedding.local_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_settings.return_value = settings
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        result = generate_embeddings(["hello"])
        mock_st_cls.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2")
        assert result == [[0.1, 0.2, 0.3]]

    @patch("src.embeddings.embedder.get_settings")
    def test_unknown_provider_raises(self, mock_settings):
        settings = MagicMock()
        settings.embedding.provider = "bedrock"
        mock_settings.return_value = settings
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            generate_embeddings(["hello"])


class TestEmbedChunks:
    def _make_chunk(self, content="test", page_num=1, chunk_type="text", section_heading=None):
        chunk = MagicMock()
        chunk.id = uuid.uuid4()
        chunk.content = content
        chunk.page_num = page_num
        chunk.chunk_type = chunk_type
        chunk.section_heading = section_heading
        chunk.embedding_id = None
        return chunk

    @patch("src.embeddings.embedder.generate_embeddings")
    @patch("src.embeddings.embedder.get_settings")
    def test_full_flow(self, mock_settings, mock_gen):
        mock_settings.return_value = MagicMock(embedding=MagicMock(batch_size=100))
        mock_gen.return_value = [[0.1, 0.2], [0.3, 0.4]]

        chunks = [self._make_chunk("c1"), self._make_chunk("c2")]
        store = MagicMock()

        count, warnings = embed_chunks(chunks, "doc-1", store)

        assert count == 2
        assert warnings == []
        store.add.assert_called_once()
        assert chunks[0].embedding_id == str(chunks[0].id)
        assert chunks[1].embedding_id == str(chunks[1].id)

    @patch("src.embeddings.embedder.generate_embeddings")
    @patch("src.embeddings.embedder.get_settings")
    def test_empty_input(self, mock_settings, mock_gen):
        mock_settings.return_value = MagicMock(embedding=MagicMock(batch_size=100))

        count, warnings = embed_chunks([], "doc-1", MagicMock())

        assert count == 0
        assert warnings == []
        mock_gen.assert_not_called()

    @patch("src.embeddings.embedder.generate_embeddings")
    @patch("src.embeddings.embedder.get_settings")
    def test_soft_failure_per_batch(self, mock_settings, mock_gen):
        mock_settings.return_value = MagicMock(embedding=MagicMock(batch_size=1))
        mock_gen.side_effect = [
            [[0.1]],
            RuntimeError("API error"),
            [[0.3]],
        ]

        chunks = [self._make_chunk("c1"), self._make_chunk("c2"), self._make_chunk("c3")]
        store = MagicMock()

        count, warnings = embed_chunks(chunks, "doc-1", store)

        assert count == 2
        assert len(warnings) == 1
        assert "API error" in warnings[0]
        # First and third chunk embedded, second not
        assert chunks[0].embedding_id == str(chunks[0].id)
        assert chunks[1].embedding_id is None
        assert chunks[2].embedding_id == str(chunks[2].id)

    @patch("src.embeddings.embedder.generate_embeddings")
    @patch("src.embeddings.embedder.get_settings")
    def test_metadata_none_section_heading(self, mock_settings, mock_gen):
        mock_settings.return_value = MagicMock(embedding=MagicMock(batch_size=100))
        mock_gen.return_value = [[0.1]]

        chunk = self._make_chunk(section_heading=None)
        store = MagicMock()

        embed_chunks([chunk], "doc-1", store)

        call_args = store.add.call_args
        metadatas = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas")
        # None should be converted to "" for ChromaDB compatibility
        assert metadatas[0]["section_heading"] == ""

    @patch("src.embeddings.embedder.generate_embeddings")
    @patch("src.embeddings.embedder.get_settings")
    def test_metadata_fields(self, mock_settings, mock_gen):
        mock_settings.return_value = MagicMock(embedding=MagicMock(batch_size=100))
        mock_gen.return_value = [[0.1]]

        chunk = self._make_chunk(page_num=3, chunk_type="table", section_heading="Results")
        store = MagicMock()

        embed_chunks([chunk], "doc-42", store)

        call_args = store.add.call_args
        metadatas = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas")
        meta = metadatas[0]
        assert meta["doc_id"] == "doc-42"
        assert meta["page_num"] == 3
        assert meta["chunk_type"] == "table"
        assert meta["section_heading"] == "Results"
