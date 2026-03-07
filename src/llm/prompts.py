"""Prompt templates for grounded RAG Q&A.

Context chunks are numbered [1], [2], ... by the context assembler.
The system prompt instructs the model to cite using those indices.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a document intelligence assistant. Answer the user's question using \
ONLY the information found in the provided document context.

Citation rules (mandatory):
- Every factual claim must be supported by a citation in the form [N], where N \
is the reference number shown at the start of each context excerpt.
- If multiple excerpts support a claim, cite all relevant ones, e.g. [1][3].
- Place citations immediately after the sentence or clause they support.

Scope rules:
- Do not use knowledge outside the provided context.
- If the context does not contain sufficient information to answer the question, \
respond exactly: "I don't have enough information in the provided documents to \
answer this question."
- Do not speculate or infer beyond what the context states.

Formatting:
- Be concise and precise.
- Prefer direct quotations when they are short and exact.
- Structure longer answers with clear paragraphs.\
"""

_NO_CONTEXT_ANSWER = (
    "I don't have enough information in the provided documents to answer this question."
)


def build_user_message(query: str, context_text: str) -> str:
    """Combine retrieved context and the user's question into a single user turn."""
    return f"Document Context:\n{context_text}\n\nQuestion: {query}"


def build_messages(
    query: str,
    context_text: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """Build the messages list for the Anthropic API.

    History turns are passed as-is (raw user questions + assistant answers
    without context, since context is retrieved fresh per turn).
    The current query is the final user message and includes the retrieved context.

    Args:
        query: Current user query.
        context_text: Assembled context string from the retrieval pipeline.
        history: Prior conversation turns as [{"role": ..., "content": ...}].

    Returns:
        Messages list ready for anthropic.messages.create().
    """
    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": build_user_message(query, context_text)})
    return messages
