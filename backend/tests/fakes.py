"""Scripted stand-in for the Anthropic client, shaped like SDK responses.

Agents touch only `.content` blocks (`.type`, `.text`, `.id`, `.name`, `.input`),
`.stop_reason` and `.parsed_output`, so plain namespaces are sufficient.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

_ids = itertools.count(1)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_call(name: str, arguments: dict[str, Any], rationale: str = "") -> SimpleNamespace:
    content = [text_block(rationale)] if rationale else []
    content.append(
        SimpleNamespace(type="tool_use", id=f"tu_{next(_ids)}", name=name, input=arguments)
    )
    return SimpleNamespace(content=content, stop_reason="tool_use")


def calls(*blocks: SimpleNamespace) -> SimpleNamespace:
    """Several tool calls in one assistant turn."""
    content = [b for r in blocks for b in r.content]
    return SimpleNamespace(content=content, stop_reason="tool_use")


def final_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[text_block(text)], stop_reason="end_turn")


def verdict(status: str, confidence: float = 0.9, reason: str = "") -> SimpleNamespace:
    return tool_call(
        "submit_verdict",
        {
            "status": status,
            "confidence": confidence,
            "summary": f"{status} after diagnostics",
            "reading": f"The motor was checked and the finding is {status}.",
            "discard_reason": reason,
        },
    )


def _blocks(content: Any) -> list[Any]:
    return content if isinstance(content, list) else [{"type": "text", "text": content}]


def _field(block: Any, name: str) -> Any:
    return block[name] if isinstance(block, dict) else getattr(block, name)


def validate_request(kwargs: dict[str, Any]) -> None:
    """Assert the request obeys the Messages API contract the real endpoint enforces."""
    for key in ("max_tokens", "messages"):
        assert key in kwargs, f"missing {key}"
    for tool in kwargs.get("tools", []):
        assert {"name", "description", "input_schema"} <= set(tool), tool
        if tool.get("strict"):
            schema = tool["input_schema"]
            assert schema.get("additionalProperties") is False, tool["name"]
            assert set(schema["required"]) == set(schema["properties"]), tool["name"]
    messages = kwargs["messages"]
    assert messages and messages[0]["role"] == "user", "first message must be user"
    pending: set[str] = set()
    previous_role = None
    for msg in messages:
        role = msg["role"]
        blocks = _blocks(msg["content"])
        if role == "assistant":
            # consecutive same-role messages are merged into one turn by the API (pause_turn)
            new = {_field(b, "id") for b in blocks if _field(b, "type") == "tool_use"}
            pending = pending | new if previous_role == "assistant" else new
        else:
            results = [b for b in blocks if _field(b, "type") == "tool_result"]
            ids = {_field(b, "tool_use_id") for b in results}
            assert ids == pending, f"tool_results {ids} must answer tool_uses {pending}"
            if results:
                first_other = next(
                    (i for i, b in enumerate(blocks) if _field(b, "type") != "tool_result"),
                    len(blocks),
                )
                assert all(_field(b, "type") != "tool_result" for b in blocks[first_other:]), (
                    "tool_result blocks must come first in the user message"
                )
            pending = set()
        previous_role = role


Script = list[SimpleNamespace] | Callable[[list[dict]], SimpleNamespace]


class ScriptedLLM:
    """Returns scripted `create` responses in order; `parse` returns a fixed object."""

    def __init__(self, script: Script | None = None, parsed: Any = None) -> None:
        self.model = "scripted-fake"
        self._script = script or []
        self._parsed = parsed
        self.create_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        validate_request(kwargs)
        # snapshot: the agent keeps appending to the same messages list
        self.create_calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if callable(self._script):
            return self._script(kwargs["messages"])
        if not self._script:
            return final_text("nothing further")
        return self._script.pop(0)

    def parse(self, output_format: type, **kwargs: Any) -> SimpleNamespace:
        self.parse_calls.append({"output_format": output_format, **kwargs})
        return SimpleNamespace(parsed_output=self._parsed)

    def tool_results(self, call_index: int) -> list[dict[str, Any]]:
        """tool_result blocks sent back to the model on a given create call."""
        last = self.create_calls[call_index]["messages"][-1]
        return [b for b in last["content"] if isinstance(b, dict) and b["type"] == "tool_result"]
