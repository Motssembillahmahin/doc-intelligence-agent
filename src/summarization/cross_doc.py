"""Cross-document summarization — synthesize summaries from multiple documents.

When more than max_chunks_per_reduce_batch documents are provided, an
intermediate reduce pass is applied before the final synthesis call.
"""

from __future__ import annotations

import structlog

from src.config import get_settings
from src.llm.client import complete
from src.summarization.prompts import CROSS_DOC_SYSTEM_PROMPT, REDUCE_SYSTEM_PROMPT

logger = structlog.get_logger(__name__)


def _format_doc_block(doc_name: str, summary: str, index: int) -> str:
    return f"[Document {index}: {doc_name}]\n{summary}"


def summarize_across_documents(doc_summaries: list[tuple[str, str]]) -> str:
    """Synthesize summaries from multiple documents into a cross-document overview.

    Args:
        doc_summaries: List of (document_filename, summary) tuples.

    Returns:
        Cross-document synthesis string, or "" if no summaries provided.
    """
    if not doc_summaries:
        return ""

    if len(doc_summaries) == 1:
        return doc_summaries[0][1]

    settings = get_settings()
    s = settings.summarization
    log = logger.bind(doc_count=len(doc_summaries))

    formatted = [
        _format_doc_block(name, summary, i + 1) for i, (name, summary) in enumerate(doc_summaries)
    ]

    # If more docs than batch size, reduce intermediately first
    batch_size = s.max_chunks_per_reduce_batch
    while len(formatted) > batch_size:
        batches = [formatted[i : i + batch_size] for i in range(0, len(formatted), batch_size)]
        intermediate: list[str] = []
        for batch in batches:
            combined = "\n\n".join(batch)
            response = complete(
                messages=[{"role": "user", "content": combined}],
                system=REDUCE_SYSTEM_PROMPT,
                max_tokens=s.reduce_max_tokens,
            )
            intermediate.append(response.content[0].text.strip())
        formatted = intermediate

    # Final cross-document synthesis
    combined = "\n\n".join(formatted)
    response = complete(
        messages=[{"role": "user", "content": combined}],
        system=CROSS_DOC_SYSTEM_PROMPT,
        max_tokens=s.reduce_max_tokens,
    )
    result = response.content[0].text.strip()

    log.info("cross_doc_summarization_complete", summary_length=len(result))
    return result
