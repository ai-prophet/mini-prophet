from __future__ import annotations

import asyncio
import json

import pytest

from miniprophet.exceptions import FormatError
from miniprophet.models.openrouter import (
    OpenRouterAPIError,
    OpenRouterAuthenticationError,
    OpenRouterModel,
    OpenRouterRateLimitError,
)


def test_openrouter_prepare_messages_strips_extra() -> None:
    m = OpenRouterModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    out = m._prepare_messages([{"role": "user", "content": "x", "extra": {"a": 1}}])
    assert out == [{"role": "user", "content": "x"}]


@pytest.mark.parametrize(
    "response,expected_cost",
    [
        ({"usage": {"cost": 0.12}}, 0.12),
        ({"usage": {"cost": None}}, 0.0),
    ],
)
def test_openrouter_calculate_cost_ok(response: dict, expected_cost: float) -> None:
    m = OpenRouterModel(model_name="free-model", cost_tracking="default")
    assert m._calculate_cost(response)["cost"] == pytest.approx(expected_cost)


def test_openrouter_calculate_cost_raises_when_missing_nonfree() -> None:
    m = OpenRouterModel(model_name="openai/gpt-4o-mini", cost_tracking="default")
    with pytest.raises(RuntimeError, match="No valid cost info"):
        m._calculate_cost({"usage": {"cost": 0.0}})


def test_openrouter_parse_actions_requires_single_call() -> None:
    m = OpenRouterModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    with pytest.raises(FormatError) as exc:
        m._parse_actions({"choices": [{"message": {"tool_calls": []}}]})
    assert "No tool call found" in exc.value.messages[0]["content"]
    assert exc.value.messages[0]["extra"]["interrupt_type"] == "FormatError"

    with pytest.raises(FormatError) as exc:
        m._parse_actions(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "1", "function": {"name": "a", "arguments": "{}"}},
                                {"id": "2", "function": {"name": "b", "arguments": "{}"}},
                            ]
                        }
                    }
                ]
            }
        )
    assert "Multiple tool calls found" in exc.value.messages[0]["content"]
    assert exc.value.messages[0]["extra"]["interrupt_type"] == "FormatError"


def test_openrouter_format_observation_messages_tool_and_user_roles() -> None:
    m = OpenRouterModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    message = {
        "extra": {
            "actions": [
                {"name": "search", "arguments": "{}", "tool_call_id": "tc1"},
                {"name": "x", "arguments": "{}", "tool_call_id": None},
            ]
        }
    }
    outputs = [{"output": "ok"}, {"output": "bad", "error": True}]
    msgs = m.format_observation_messages(message, outputs)

    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "tc1"
    assert "<output>" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "<error>" in msgs[1]["content"]


class _FakeHttpxResponse:
    def __init__(self, status_code: int, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = json.dumps(json_data) if json_data else "error text"

    def json(self):
        return self._json_data

    def raise_for_status(self):
        import httpx

        if self.status_code >= 400:
            request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            response = httpx.Response(self.status_code, text=self.text, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


@pytest.mark.parametrize(
    "status_code,exc_type",
    [
        (401, OpenRouterAuthenticationError),
        (429, OpenRouterRateLimitError),
        (500, OpenRouterAPIError),
    ],
)
def test_openrouter_query_http_errors(monkeypatch, status_code: int, exc_type: type) -> None:
    import httpx

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    m = OpenRouterModel(model_name="test/model", cost_tracking="ignore_errors")

    async def fake_post(self, url, **kwargs):
        resp = _FakeHttpxResponse(status_code)
        resp.raise_for_status()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(exc_type):
        asyncio.run(m._query([], []))


def test_openrouter_query_connection_error(monkeypatch) -> None:
    import httpx

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    m = OpenRouterModel(model_name="test/model", cost_tracking="ignore_errors")

    async def fake_post(self, url, **kwargs):
        raise httpx.RequestError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(OpenRouterAPIError, match="Request failed"):
        asyncio.run(m._query([], []))
