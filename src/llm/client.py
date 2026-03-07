"""Anthropic client — lazy singleton wrapper.

All LLM calls go through `complete()`. The client is instantiated once per
process so connection overhead is paid only on the first call.
"""

from __future__ import annotations

import anthropic
import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        settings = get_settings()
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        logger.info("anthropic_client_initialized", model=settings.llm.model)
    return _client


def complete(
    messages: list[dict],
    system: str,
    max_tokens: int | None = None,
) -> anthropic.types.Message:
    """Send a chat completion request to the Anthropic API.

    Args:
        messages: Conversation turns as [{"role": "user"|"assistant", "content": str}].
        system: The system prompt string.
        max_tokens: Override for the output token limit. Defaults to llm.max_tokens
                    from settings. Pass a lower value for short summarization steps.

    Returns:
        The raw anthropic.types.Message response (caller accesses .content and .usage).
    """
    settings = get_settings()
    client = _get_client()

    response = client.messages.create(
        model=settings.llm.model,
        max_tokens=max_tokens if max_tokens is not None else settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        system=system,
        messages=messages,
    )

    logger.info(
        "llm_response_received",
        model=settings.llm.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        stop_reason=response.stop_reason,
    )
    return response
