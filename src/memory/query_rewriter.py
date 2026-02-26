"""Query rewriter — resolves pronoun references against conversation history.

When a user asks "What else does it say?" or "Summarize that section", the
rewriter expands the query into a self-contained question that retrieval can
act on without needing the conversation context.
"""

from __future__ import annotations

import structlog

from src.llm.client import complete

logger = structlog.get_logger(__name__)

_REWRITE_SYSTEM_PROMPT = """\
Given a conversation history and a follow-up question, rewrite the follow-up question \
to be fully self-contained and understandable without the conversation context.
Resolve all pronouns, references, and ellipses using information from the history.
If the question is already self-contained, return it exactly as written.
Output ONLY the rewritten question — no explanation, no preamble.\
"""


def rewrite_query(query: str, history: list[dict]) -> str:
    """Return a self-contained version of query given the conversation history.

    If history is empty the original query is returned unchanged (no API call).

    Args:
        query: The raw user query that may contain unresolved references.
        history: Anthropic-format prior turns [{"role": ..., "content": ...}].

    Returns:
        The rewritten query, or the original query if rewriting fails or
        history is empty.
    """
    if not history:
        return query

    # Build a compact transcript from the last few turns (avoid large prompts)
    recent = history[-6:]
    transcript = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in recent)
    prompt = f"Conversation history:\n{transcript}\n\nFollow-up question: {query}"

    try:
        response = complete(
            messages=[{"role": "user", "content": prompt}],
            system=_REWRITE_SYSTEM_PROMPT,
            max_tokens=200,
        )
        rewritten = response.content[0].text.strip()
    except Exception as exc:
        logger.warning("query_rewrite_failed", error=str(exc), query=query[:80])
        return query

    if not rewritten:
        return query

    if rewritten != query:
        logger.info("query_rewritten", original=query[:80], rewritten=rewritten[:80])

    return rewritten
