"""Metric persistence — lightweight time-series counters stored in PostgreSQL."""

from __future__ import annotations

import structlog
from sqlmodel import Session

from src.models.database import Metric

logger = structlog.get_logger(__name__)


def record_metric(
    name: str,
    value: float,
    db_session: Session,
    labels: dict | None = None,
) -> None:
    """Persist a single metric data point.

    Soft-fails on error so that metric recording never breaks the main flow.

    Args:
        name: Dot-separated metric name, e.g. "chat.input_tokens".
        value: Numeric value to record.
        db_session: Synchronous DB session.
        labels: Optional key-value tags (stored as JSONB).
    """
    try:
        db_session.add(Metric(metric_name=name, metric_value=value, labels=labels))
        db_session.commit()
    except Exception as exc:
        logger.warning("metric_record_failed", metric=name, error=str(exc))
        db_session.rollback()
