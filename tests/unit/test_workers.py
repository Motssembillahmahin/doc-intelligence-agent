"""Unit tests for Celery tasks."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestIngestDocumentTask:
    @patch("src.services.ingestion_service.run_ingestion")
    @patch("src.services.ingestion_service.get_sync_session")
    def test_task_calls_service(self, mock_session_fn, mock_pipeline):
        # Set up mock DB session
        doc_id = uuid.uuid4()
        mock_doc = MagicMock()
        mock_doc.id = doc_id

        session = MagicMock()
        session.get.return_value = mock_doc
        mock_session_fn.return_value = session

        # Set up mock pipeline result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.validation.page_count = 3
        mock_result.warnings = []
        mock_result.error = None
        mock_pipeline.return_value = mock_result

        # Call the service function directly (not through Celery)
        from src.services.ingestion_service import ingest_document

        result = ingest_document(doc_id, Path("/tmp/test.pdf"))

        assert result.success is True
        mock_pipeline.assert_called_once()
