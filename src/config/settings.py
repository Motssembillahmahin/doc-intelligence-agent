from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, YamlConfigSettingsSource

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class DatabaseSettings(BaseModel):
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


class IngestionSettings(BaseModel):
    max_file_size_mb: int = 50
    supported_formats: list[str] = [".pdf"]
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    image_dpi: int = 300


class ChunkingSettings(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n\n"
    tiktoken_encoding: str = "cl100k_base"


class EmbeddingSettings(BaseModel):
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100


class RetrievalSettings(BaseModel):
    top_k: int = 10
    rerank_top_k: int = 5
    alpha: float = 0.7
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_context_tokens: int = 8000


class LLMSettings(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.1


class ConversationSettings(BaseModel):
    max_history_turns: int = 5
    summary_after_turns: int = 10


class ProcessingSettings(BaseModel):
    celery_task_timeout: int = 600
    max_concurrent_tasks: int = 4


class ObservabilitySettings(BaseModel):
    log_level: str = "INFO"
    log_format: str = "json"
    enable_traces: bool = True


class ChromaDBSettings(BaseModel):
    collection_name: str = "doc_chunks"
    distance_function: str = "cosine"


class Settings(BaseSettings):
    # Secrets from .env
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    database_url: str = "postgresql+asyncpg://docagent:docagent@localhost:5432/docagent"
    database_url_sync: str = "postgresql+psycopg2://docagent:docagent@localhost:5432/docagent"
    celery_broker_url: str = "sqla+postgresql://docagent:docagent@localhost:5432/docagent"
    celery_result_backend: str = "db+postgresql://docagent:docagent@localhost:5432/docagent"
    chroma_host: str = "localhost"
    chroma_port: int = 8100
    app_env: str = "dev"

    # Nested settings groups from vars.yaml
    database: DatabaseSettings = DatabaseSettings()
    ingestion: IngestionSettings = IngestionSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    llm: LLMSettings = LLMSettings()
    conversation: ConversationSettings = ConversationSettings()
    processing: ProcessingSettings = ProcessingSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    chromadb: ChromaDBSettings = ChromaDBSettings()

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_nested_delimiter": "__",
        "extra": "ignore",
    }

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        app_env = os.getenv("APP_ENV", "dev")
        yaml_files = [str(CONFIG_DIR / "vars.yaml")]
        env_yaml = CONFIG_DIR / f"vars.{app_env}.yaml"
        if env_yaml.exists():
            yaml_files.append(str(env_yaml))

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(
                settings_cls,
                yaml_file=yaml_files,
            ),
            file_secret_settings,
        )
