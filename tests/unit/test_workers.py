"""Unit tests for Celery tasks."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.workers.tasks as tasks_module
from src.workers.tasks import ingest_document_task


class TestIngestDocumentTask:
    def test_task_is_registered(self):
        assert ingest_document_task.name == "ingest_document"
        assert ingest_document_task.max_retries == 1

    def test_task_calls_service_and_returns_result(self):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.validation.page_count = 3
        mock_result.warnings = []
        mock_result.error = None

        doc_id = str(uuid.uuid4())

        with patch.object(
            tasks_module, "ingest_document", return_value=mock_result
        ) as mock_service:
            result = ingest_document_task.run(doc_id, "/tmp/test.pdf")

        mock_service.assert_called_once_with(uuid.UUID(doc_id), Path("/tmp/test.pdf"))
        assert result["success"] is True
        assert result["doc_id"] == doc_id
        assert result["page_count"] == 3

    def test_task_calls_retry_on_exception(self):
        doc_id = str(uuid.uuid4())

        with (
            patch.object(
                tasks_module, "ingest_document", side_effect=RuntimeError("connection lost")
            ),
            patch.object(
                ingest_document_task, "retry", side_effect=RuntimeError("retrying")
            ) as mock_retry,
            pytest.raises(RuntimeError, match="retrying"),
        ):
            ingest_document_task.run(doc_id, "/tmp/test.pdf")

        mock_retry.assert_called_once()
