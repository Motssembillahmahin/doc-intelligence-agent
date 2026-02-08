"""Unit tests for the ingestion service."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.database import Document
from src.models.enums import DocumentStatus
from src.services.ingestion_service import create_document_record, ingest_document


class TestIngestDocument:
    @patch("src.services.ingestion_service.get_sync_session")
    @patch("src.services.ingestion_service.run_ingestion")
    def test_successful_ingestion(self, mock_pipeline, mock_session_fn):
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
        mock_pipeline.return_value = mock_result

        result = ingest_document(doc_id, Path("/tmp/test.pdf"))

        assert result.success is True
        assert mock_doc.status == DocumentStatus.completed
        assert session.commit.call_count == 2  # processing + completed

    @patch("src.services.ingestion_service.get_sync_session")
    @patch("src.services.ingestion_service.run_ingestion")
    def test_ingestion_with_warnings(self, mock_pipeline, mock_session_fn):
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
        mock_pipeline.return_value = mock_result

        ingest_document(doc_id, Path("/tmp/test.pdf"))

        assert mock_doc.status == DocumentStatus.completed_with_warnings
        assert mock_doc.warnings == ["OCR failed on page 2"]

    @patch("src.services.ingestion_service.get_sync_session")
    @patch("src.services.ingestion_service.run_ingestion")
    def test_pipeline_failure(self, mock_pipeline, mock_session_fn):
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
