"""LLM client boundary.

Agents depend on the `LLMClient` Protocol, not on the SDK, so tests drive them with a scripted
fake and deployment swaps providers in one place. `AnthropicLLM` is the real implementation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"


class AgentUnavailableError(RuntimeError):
    """The LLM backend is not configured or not reachable."""


class LLMClient(Protocol):
    model: str

    def create(self, **kwargs: Any) -> Any:
        """messages.create: returns an object with .content (blocks) and .stop_reason."""
        ...

    def parse(self, output_format: type, **kwargs: Any) -> Any:
        """messages.parse: returns an object with .parsed_output (validated model)."""
        ...


class AnthropicLLM:
    """Anthropic SDK adapter that maps SDK errors onto AgentUnavailableError."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        import anthropic

        self._anthropic = anthropic
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def _call(self, fn, **kwargs: Any) -> Any:
        a = self._anthropic
        try:
            return fn(model=self.model, **kwargs)
        except a.AuthenticationError as err:
            raise AgentUnavailableError("Anthropic API rejected the credentials") from err
        except a.RateLimitError as err:
            raise AgentUnavailableError("Anthropic API rate limit reached; retry later") from err
        except a.APIConnectionError as err:
            raise AgentUnavailableError("Cannot reach the Anthropic API") from err
        except a.APIStatusError as err:
            logger.error("Anthropic API error %s: %s", err.status_code, err.message)
            raise AgentUnavailableError(f"Anthropic API error ({err.status_code})") from err

    def create(self, **kwargs: Any) -> Any:
        return self._call(self._client.messages.create, **kwargs)

    def parse(self, output_format: type, **kwargs: Any) -> Any:
        return self._call(self._client.messages.parse, output_format=output_format, **kwargs)


def llm_from_env() -> LLMClient | None:
    """The configured client, or None when no API key is set (agents then report 503)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return AnthropicLLM()
