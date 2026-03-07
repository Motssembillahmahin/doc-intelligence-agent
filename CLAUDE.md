# CLAUDE.md — Project Instructions

## Project Overview
AI-Powered Document Intelligence System — RAG-based platform that ingests PDFs, extracts content (text, tables, images, OCR), generates summaries, and supports grounded conversational Q&A with citations.

## Tech Stack
- **Language**: Python 3.11+
- **API**: FastAPI
- **ORM**: SQLModel (Pydantic + SQLAlchemy)
- **Database**: PostgreSQL (state, traces, metrics, Celery broker+backend, FTS)
- **Vector Store**: ChromaDB (behind abstraction layer)
- **LLM**: Anthropic Claude (via SDK)
- **Embeddings**: OpenAI text-embedding-3-small (configurable)
- **PDF Parsing**: PyMuPDF, pdfplumber, Camelot, pytesseract
- **Chunking**: LangChain RecursiveCharacterTextSplitter + tiktoken
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Background Jobs**: Celery (PostgreSQL broker+backend)
- **Logging**: structlog
- **UI**: Streamlit

## Tooling
- **Formatter/Linter**: Ruff (`ruff check --fix . && ruff format .`)
- **Package Manager**: uv
- **Task Runner**: Makefile
- **Testing**: pytest + pytest-asyncio

## Project Structure
```
src/
  config/          # Pydantic Settings, loads vars.yaml + .env
  models/          # SQLModel DB models + Pydantic schemas
  db/              # Engine, session factory, Alembic migrations
  ingestion/       # PDF parsing, OCR, table extraction, image processing
  chunking/        # Heading detection, text splitting
  embeddings/      # Embedding generation, vector store protocol + ChromaDB impl
  retrieval/       # Hybrid search, reranking, context assembly
  llm/             # Anthropic client, prompts, relevance checking
  summarization/   # Map-reduce per-doc + cross-doc
  memory/          # Conversation sessions, query rewriting, history management
  workers/         # Celery app + task definitions
  services/        # Business logic layer (between API routes and modules)
  api/             # FastAPI app, routes (documents, chat, summaries)
  ui/              # Streamlit app
  observability/   # structlog config, traces, metrics
tests/
  unit/            # Component-level tests
  integration/     # Pipeline tests
  eval/            # RAG quality evaluation
  fixtures/pdfs/   # Test PDFs (generated + manual)
config/
  vars.yaml        # All tunables (version controlled)
data/
  pdfs/            # User documents
  images/          # Extracted images
```

## Key Commands
```bash
make dev           # Start PostgreSQL + ChromaDB via docker-compose
make migrate       # Run Alembic migrations
make api           # Start FastAPI server
make worker        # Start Celery worker
make ui            # Start Streamlit
make test          # Run pytest
make lint          # Run ruff check + format
```

## Code Conventions
- **Services layer**: All business logic in `src/services/`. API routes and Celery tasks are thin wrappers.
- **Config**: Secrets in `.env` (gitignored), tunables in `config/vars.yaml`. Access via `src/config/settings.py`.
- **Models**: SQLModel for DB models. Pydantic for API request/response schemas.
- **Logging**: Use structlog. Always include contextual fields (doc_id, session_id, etc.).
- **Formatting**: Run `ruff check --fix . && ruff format .` before committing.
- **Type hints**: Use throughout. SQLModel/Pydantic enforce at runtime.
- **Imports**: Absolute imports from `src.*`.
- **Error handling**: Document-level tracking. Soft failures → completed_with_warnings. Hard failures → failed.
- **Testing**: pytest. Unit tests for each module, integration tests for pipelines.

## Architecture Rules
- Never put business logic in API routes or Celery tasks
- Vector store access only through the abstraction layer (protocol/interface)
- All tunables must come from vars.yaml via Settings, never hardcoded
- Page-by-page streaming for PDF processing (never load full doc in memory)
- Event-driven ingestion: upload triggers background Celery job
- Hybrid search always: vector + PostgreSQL FTS, fused with configurable alpha
- Every chunk carries metadata: doc_id, page_num, chunk_type, section_heading
- LLM responses must include citation instructions in system prompt

## Git Workflow
- Branch: `dev` for development, `main` for stable
- CI: GitHub Actions runs lint + tests on every push
- Commits: Conventional style, descriptive messages
