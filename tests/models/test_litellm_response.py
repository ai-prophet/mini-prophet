from __future__ import annotations

import json
import types

import pytest

from miniprophet.exceptions import FormatError
from miniprophet.models.litellm_response import LitellmResponseModel


def _responses_obj(output_items: list[dict]) -> types.SimpleNamespace:
    response = types.SimpleNamespace(output=output_items)
    response.model_dump = lambda: {"object": "response", "output": output_items}
    return response


def test_litellm_response_prepare_messages_converts_chat_history() -> None:
    model = LitellmResponseModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    messages = [
        {"role": "user", "content": "hello", "extra": {"x": 1}},
        {
            "role": "assistant",
            "content": "working",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": {"query": "inflation"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
    ]

    prepared = model._prepare_messages(messages)

    assert prepared[0] == {"role": "user", "content": "hello"}
    assert prepared[1] == {"role": "assistant", "content": "working"}
    assert prepared[2]["type"] == "function_call"
    assert prepared[2]["call_id"] == "call_1"
    assert prepared[2]["name"] == "search"
    assert json.loads(prepared[2]["arguments"]) == {"query": "inflation"}
    assert prepared[3] == {"type": "function_call_output", "call_id": "call_1", "output": "done"}


def test_litellm_response_prepare_tools_converts_chat_tools() -> None:
    model = LitellmResponseModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search web",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {"type": "other", "x": 1},
    ]

    prepared = model._prepare_tools(tools)
    assert prepared[0] == {
        "type": "function",
        "name": "search",
        "description": "Search web",
        "parameters": {"type": "object", "properties": {}},
    }
    assert prepared[1] == {"type": "other", "x": 1}


def test_litellm_response_parse_actions_requires_single_call() -> None:
    model = LitellmResponseModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")

    with pytest.raises(FormatError) as exc:
        model._parse_actions(_responses_obj([]))
    assert "No tool call found" in exc.value.messages[0]["content"]

    with pytest.raises(FormatError) as exc:
        model._parse_actions(
            _responses_obj(
                [
                    {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
                    {"type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}"},
                ]
            )
        )
    assert "Multiple tool calls found" in exc.value.messages[0]["content"]


def test_litellm_response_build_message_from_response_output() -> None:
    model = LitellmResponseModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    response = _responses_obj(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Thinking"}],
            },
            {
                "type": "function_call",
                "call_id": "call_7",
                "name": "search",
                "arguments": {"query": "latest inflation"},
            },
        ]
    )

    message = model._build_message(response)
    assert message["role"] == "assistant"
    assert message["content"] == "Thinking"
    assert message["tool_calls"][0]["id"] == "call_7"
    assert message["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "query": "latest inflation"
    }


def test_litellm_response_query_uses_prepare_messages_hook() -> None:
    model = LitellmResponseModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")

    captured: dict[str, list[dict]] = {}
    response = _responses_obj(
        [{"type": "function_call", "call_id": "call_1", "name": "search", "arguments": "{}"}]
    )

    def fake_query(messages: list[dict], tools: list[dict]):
        captured["messages"] = messages
        captured["tools"] = tools
        return response

    model._query = fake_query  # type: ignore[method-assign]
    model._calculate_cost = lambda response: {"cost": 0.0}  # type: ignore[method-assign]

    msgs = [
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "call_1", "content": "search result"},
    ]
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    model.query(msgs, tools)

    assert captured["messages"][1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "search result",
    }
