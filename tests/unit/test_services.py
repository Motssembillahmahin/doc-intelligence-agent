"""Unit tests for the ingestion service."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.chunking.splitter import ChunkData
from src.models.database import Document
from src.models.enums import ChunkType, DocumentStatus
from src.services.ingestion_service import create_document_record, ingest_document


class TestIngestDocument:
    @patch("src.embeddings.embedder.embed_chunks")
    @patch("src.embeddings.chromadb_store.ChromaDBStore")
    @patch("src.services.ingestion_service.split_pages")
    @patch("src.services.ingestion_service._build_headings_map")
    @patch("src.services.ingestion_service.run_ingestion")
    @patch("src.services.ingestion_service.get_sync_session")
    def test_successful_ingestion(
        self, mock_session_fn, mock_pipeline, mock_headings, mock_split, mock_chroma, mock_embed
    ):
        doc_id = uuid.uuid4()
        mock_doc = MagicMock(spec=Document)
        mock_doc.id = doc_id

        session = MagicMock()
        session.get.return_value = mock_doc
        mock_session_fn.return_value = session

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.validation.page_count = 5
        mock_result.warnings = []
        mock_result.error = None
        mock_result.pages = []
        mock_pipeline.return_value = mock_result

        mock_headings.return_value = {}
        mock_split.return_value = [
            ChunkData(
                content="test chunk",
                chunk_type=ChunkType.text,
                page_num=1,
                chunk_index=0,
                section_heading=None,
                token_count=2,
            ),
        ]
        mock_embed.return_value = (1, [])

        result = ingest_document(doc_id, Path("/tmp/test.pdf"))

        assert result.success is True
        assert mock_doc.status == DocumentStatus.completed
        assert session.commit.call_count == 2  # processing + completed
        # Chunk was added to session
        assert session.add.call_count >= 1
        mock_split.assert_called_once()
        mock_embed.assert_called_once()

    @patch("src.services.ingestion_service.split_pages")
    @patch("src.services.ingestion_service._build_headings_map")
    @patch("src.services.ingestion_service.run_ingestion")
    @patch("src.services.ingestion_service.get_sync_session")
    def test_ingestion_with_warnings(
        self, mock_session_fn, mock_pipeline, mock_headings, mock_split
    ):
        doc_id = uuid.uuid4()
        mock_doc = MagicMock(spec=Document)

        session = MagicMock()
        session.get.return_value = mock_doc
        mock_session_fn.return_value = session

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.validation.page_count = 3
        mock_result.warnings = ["OCR failed on page 2"]
        mock_result.error = None
        mock_result.pages = []
        mock_pipeline.return_value = mock_result

        mock_headings.return_value = {}
        mock_split.return_value = []  # No chunks → embedding step skipped

        ingest_document(doc_id, Path("/tmp/test.pdf"))

        assert mock_doc.status == DocumentStatus.completed_with_warnings
        assert mock_doc.warnings == ["OCR failed on page 2"]

    @patch("src.services.ingestion_service.run_ingestion")
    @patch("src.services.ingestion_service.get_sync_session")
    def test_pipeline_failure(self, mock_session_fn, mock_pipeline):
        doc_id = uuid.uuid4()
        mock_doc = MagicMock(spec=Document)

        session = MagicMock()
        session.get.return_value = mock_doc
        mock_session_fn.return_value = session

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "PDF corrupt"
        mock_pipeline.return_value = mock_result

        ingest_document(doc_id, Path("/tmp/test.pdf"))

        assert mock_doc.status == DocumentStatus.failed
        assert mock_doc.error_message == "PDF corrupt"

    @patch("src.services.ingestion_service.get_sync_session")
    def test_document_not_found(self, mock_session_fn):
        session = MagicMock()
        session.get.return_value = None
        mock_session_fn.return_value = session

        with pytest.raises(ValueError, match="not found"):
            ingest_document(uuid.uuid4(), Path("/tmp/test.pdf"))

    @patch("src.embeddings.embedder.embed_chunks")
    @patch("src.embeddings.chromadb_store.ChromaDBStore")
    @patch("src.services.ingestion_service.split_pages")
    @patch("src.services.ingestion_service._build_headings_map")
    @patch("src.services.ingestion_service.run_ingestion")
    @patch("src.services.ingestion_service.get_sync_session")
    def test_chunks_persisted_to_db(
        self, mock_session_fn, mock_pipeline, mock_headings, mock_split, mock_chroma, mock_embed
    ):
        doc_id = uuid.uuid4()
        mock_doc = MagicMock(spec=Document)
        mock_doc.id = doc_id

        session = MagicMock()
        session.get.return_value = mock_doc
        mock_session_fn.return_value = session

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.validation.page_count = 1
        mock_result.warnings = []
        mock_result.error = None
        mock_result.pages = []
        mock_pipeline.return_value = mock_result
        mock_embed.return_value = (2, [])

        mock_headings.return_value = {1: "Introduction"}
        mock_split.return_value = [
            ChunkData(
                content="chunk one",
                chunk_type=ChunkType.text,
                page_num=1,
                chunk_index=0,
                section_heading="Introduction",
                token_count=2,
            ),
            ChunkData(
                content="| A | B |",
                chunk_type=ChunkType.table,
                page_num=1,
                chunk_index=1,
                section_heading="Introduction",
                token_count=5,
                metadata={"accuracy": 95.0, "method": "lattice"},
            ),
        ]

        ingest_document(doc_id, Path("/tmp/test.pdf"))

        # 2 chunks added to session
        add_calls = [c for c in session.add.call_args_list]
        assert len(add_calls) == 2


class TestCreateDocumentRecord:
    @patch("src.services.ingestion_service.get_sync_session")
    def test_creates_document(self, mock_session_fn):
        session = MagicMock()
        mock_session_fn.return_value = session

        create_document_record(
            filename="test.pdf",
            file_path=Path("/tmp/test.pdf"),
            file_hash="abc123",
            file_size_bytes=1024,
        )

        session.add.assert_called_once()
        session.commit.assert_called_once()
        session.refresh.assert_called_once()
