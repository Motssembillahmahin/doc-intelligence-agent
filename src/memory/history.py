"""History management — message persistence, retrieval, and compression.

History is returned as Anthropic-format message dicts for direct use in
build_messages(). When a session has a compressed summary, it is injected
as a synthetic prior turn so the LLM is aware of earlier context.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlmodel import Session as DBSession
from sqlmodel import func, select

from src.config import get_settings
from src.models.database import Message, Session
from src.models.enums import MessageRole

logger = structlog.get_logger(__name__)

_COMPRESSION_SYSTEM_PROMPT = """\
You are a conversation memory assistant. Summarize the following conversation into a \
concise record (3-5 sentences) covering: the main topics discussed, key facts established, \
important conclusions reached, and any user preferences or constraints mentioned.\
"""


def save_message(
    session_id: uuid.UUID,
    role: MessageRole,
    content: str,
    db_session: DBSession,
    original_query: str | None = None,
    rewritten_query: str | None = None,
) -> Message:
    """Persist a single message to the DB and bump Session.updated_at."""
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        original_query=original_query,
        rewritten_query=rewritten_query,
    )
    db_session.add(msg)

    # Bump session timestamp so list_sessions ordering stays correct
    session = db_session.get(Session, session_id)
    if session:
        session.updated_at = datetime.now(tz=UTC)

    db_session.commit()
    return msg


def get_history(session_id: uuid.UUID, db_session: DBSession) -> list[dict]:
    """Load recent conversation turns as an Anthropic-format messages list.

    Returns the last max_history_turns*2 messages (each turn = user + assistant).
    If the session has a compressed summary, it is prepended as a synthetic turn
    so the LLM is aware of prior context that was pruned.
    """
    settings = get_settings()
    limit = settings.conversation.max_history_turns * 2

    rows = db_session.exec(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    ).all()

    # Reverse so messages are chronological (oldest → newest)
    messages = [{"role": str(m.role), "content": m.content} for m in reversed(rows)]

    # Prepend compressed summary as a synthetic prior turn
    session = db_session.get(Session, session_id)
    if session and session.summary and messages:
        messages = [
            {"role": "user", "content": "[Summary of earlier conversation]"},
            {"role": "assistant", "content": session.summary},
        ] + messages

    return messages


def _message_count(session_id: uuid.UUID, db_session: DBSession) -> int:
    result = db_session.exec(select(func.count()).where(Message.session_id == session_id)).one()
    return result


def compress_history(session_id: uuid.UUID, db_session: DBSession) -> None:
    """Compress old messages into Session.summary if the threshold is reached.

    When total message count >= summary_after_turns*2, all messages except the
    most recent max_history_turns*2 are summarized with the LLM, stored in
    Session.summary, and deleted from the messages table.
    """
    from src.llm.client import complete  # local import to avoid circular dependency

    settings = get_settings()
    threshold = settings.conversation.summary_after_turns * 2
    keep = settings.conversation.max_history_turns * 2

    total = _message_count(session_id, db_session)
    if total < threshold:
        return

    log = logger.bind(session_id=str(session_id), total=total, threshold=threshold)
    log.info("compression_triggered")

    # Load all messages chronologically
    all_messages = db_session.exec(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
    ).all()

    to_compress = all_messages[:-keep] if keep else all_messages
    if not to_compress:
        return

    # Format messages for the compression LLM call
    session = db_session.get(Session, session_id)
    parts: list[str] = []
    if session and session.summary:
        parts.append(f"Previous summary:\n{session.summary}\n")
    parts.append("Conversation:\n")
    parts.extend(f"{m.role.upper()}: {m.content}" for m in to_compress)

    transcript = "\n".join(parts)

    try:
        response = complete(
            messages=[{"role": "user", "content": transcript}],
            system=_COMPRESSION_SYSTEM_PROMPT,
            max_tokens=512,
        )
        new_summary = response.content[0].text.strip()
    except Exception as exc:
        log.warning("compression_llm_failed", error=str(exc))
        return

    # Persist new summary and delete compressed messages
    if session:
        session.summary = new_summary
        session.updated_at = datetime.now(tz=UTC)

    for msg in to_compress:
        db_session.delete(msg)

    db_session.commit()
    log.info("compression_complete", compressed=len(to_compress), summary_length=len(new_summary))
