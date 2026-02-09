import enum


class DocumentStatus(enum.StrEnum):
    """Lifecycle status of an ingested document."""

    pending = "pending"
    processing = "processing"
    completed = "completed"
    completed_with_warnings = "completed_with_warnings"
    failed = "failed"
    skipped = "skipped"


class ChunkType(enum.StrEnum):
    text = "text"
    table = "table"
    image_caption = "image_caption"


class MessageRole(enum.StrEnum):
    user = "user"
    assistant = "assistant"
