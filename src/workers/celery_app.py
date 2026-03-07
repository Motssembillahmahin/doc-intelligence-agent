from celery import Celery
from celery.signals import worker_process_init

from src.config import get_settings

settings = get_settings()

celery_app = Celery(
    "doc_intelligence",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_time_limit=settings.processing.celery_task_timeout,
    worker_max_tasks_per_child=100,
    worker_concurrency=settings.processing.max_concurrent_tasks,
)


@worker_process_init.connect
def configure_worker_logging(**kwargs: object) -> None:
    """Configure structlog once per worker process on startup."""
    from src.observability.logging import configure_logging

    configure_logging(settings.observability.log_level, settings.observability.log_format)
