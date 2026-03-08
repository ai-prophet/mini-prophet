from __future__ import annotations

import pytest

from miniprophet.exceptions import FormatError
from miniprophet.models.openrouter import OpenRouterModel


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
