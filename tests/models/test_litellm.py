from __future__ import annotations

import types

import pytest

from miniprophet.exceptions import FormatError
from miniprophet.models.litellm import LitellmModel


def _litellm_response(tool_calls):
    msg = types.SimpleNamespace(tool_calls=tool_calls)
    msg.model_dump = lambda: {
        "tool_calls": [
            {
                "id": getattr(tc, "id", None),
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", ""),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                },
            }
            for tc in (tool_calls or [])
        ]
    }
    choice = types.SimpleNamespace(message=msg)
    resp = types.SimpleNamespace(choices=[choice])
    resp.model_dump = lambda: {"choices": [{"message": msg.model_dump()}]}
    return resp


def test_litellm_parse_actions_single_call() -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    tc = types.SimpleNamespace(
        id="x",
        function=types.SimpleNamespace(name="search", arguments='{"query":"q"}'),
    )
    actions = model._parse_actions(_litellm_response([tc]))
    assert actions == [{"name": "search", "arguments": '{"query":"q"}', "tool_call_id": "x"}]


def test_litellm_parse_actions_raises_on_empty() -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    with pytest.raises(FormatError) as exc:
        model._parse_actions(_litellm_response([]))
    assert "No tool call found" in exc.value.messages[0]["content"]
    assert exc.value.messages[0]["extra"]["interrupt_type"] == "FormatError"


def test_litellm_calculate_cost_raises_when_tracking_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="default")
    monkeypatch.setattr(
        "miniprophet.models.litellm.litellm.cost_calculator.completion_cost",
        lambda response, model: 0.0,
    )
    with pytest.raises(RuntimeError, match="Error calculating cost"):
        model._calculate_cost(object())


def test_litellm_extract_usage_with_cache_tokens() -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    details = types.SimpleNamespace(cached_tokens=500, cache_creation_tokens=200)
    usage = types.SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=50,
        total_tokens=1050,
        prompt_tokens_details=details,
    )
    response = types.SimpleNamespace(usage=usage)
    result = model._extract_usage(response)
    assert result["cached_tokens"] == 500
    assert result["cache_creation_tokens"] == 200


def test_litellm_extract_usage_without_cache_details() -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    usage = types.SimpleNamespace(prompt_tokens=1000, completion_tokens=50, total_tokens=1050)
    response = types.SimpleNamespace(usage=usage)
    result = model._extract_usage(response)
    assert result["cached_tokens"] is None
    assert result["cache_creation_tokens"] is None


def test_litellm_extract_usage_with_none_cache_fields() -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    details = types.SimpleNamespace(cached_tokens=None, cache_creation_tokens=None)
    usage = types.SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=50,
        total_tokens=1050,
        prompt_tokens_details=details,
    )
    response = types.SimpleNamespace(usage=usage)
    result = model._extract_usage(response)
    assert result["cached_tokens"] is None
    assert result["cache_creation_tokens"] is None


def test_litellm_calculate_cost_ignore_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    model = LitellmModel(model_name="openai/gpt-4o-mini", cost_tracking="ignore_errors")
    monkeypatch.setattr(
        "miniprophet.models.litellm.litellm.cost_calculator.completion_cost",
        lambda response, model: (_ for _ in ()).throw(ValueError("boom")),
    )
    assert model._calculate_cost(object())["cost"] == 0.0
