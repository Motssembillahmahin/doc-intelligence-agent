.PHONY: install dev down migrate api worker ui test lint format clean

install:
	uv sync --dev

dev:
	docker compose up -d

down:
	docker compose down

migrate:
	uv run alembic upgrade head

api:
	uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

worker:
	uv run celery -A src.workers.app worker --loglevel=info

ui:
	uv run streamlit run src/ui/app.py

test:
	uv run pytest

lint:
	uv run ruff check --fix . && uv run ruff format .

format:
	uv run ruff format .

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
