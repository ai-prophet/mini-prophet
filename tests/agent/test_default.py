from __future__ import annotations

from pathlib import Path

import pytest
from conftest import DummyEnvironment, DummyModel

from miniprophet.agent.default import DefaultForecastAgent
from miniprophet.exceptions import Submitted


class _SubmitEnv(DummyEnvironment):
    async def execute(self, action: dict, **kwargs) -> dict:
        if action.get("name") == "submit":
            raise Submitted(
                {
                    "role": "exit",
                    "content": "done",
                    "extra": {
                        "exit_status": "submitted",
                        "submission": {"Yes": 0.7, "No": 0.3},
                        "board": [],
                    },
                }
            )
        return await super().execute(action, **kwargs)


def _agent_kwargs(tmp_path: Path) -> dict:
    return {
        "system_template": "sys: {title}",
        "instance_template": "inst: {title} {current_time}",
        "step_limit": 2,
        "cost_limit": 99.0,
        "search_limit": 9,
        "output_path": tmp_path,
    }


def test_default_agent_run_submitted_and_evaluated(
    assistant_action_message,
    tmp_path: Path,
) -> None:
    model = DummyModel(
        scripted_messages=[
            assistant_action_message(name="submit", arguments='{"probability": 0.7}')
        ]
    )
    env = _SubmitEnv()
    agent = DefaultForecastAgent(model=model, env=env, **_agent_kwargs(tmp_path))

    result = agent.run_sync(
        title="Will X happen?",
        ground_truth={"Yes": 1, "No": 0},
    )

    assert result["exit_status"] == "submitted"
    assert result["submission"] == {"Yes": 0.7, "No": 0.3}
    assert result["evaluation"]["brier_score"] == pytest.approx(0.09)
    assert (tmp_path / "info.json").exists()
    assert (tmp_path / "trajectory.json").exists()
    assert (tmp_path / "sources.json").exists()


class _RecordingContext:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def manage(self, messages: list[dict], *, step: int, **kwargs) -> list[dict]:
        return messages

    def record_query(self, query: str) -> None:
        self.queries.append(query)

    def display(self) -> None:
        return None


def test_default_agent_tracks_search_cost_and_query_history(
    assistant_action_message,
    tmp_path: Path,
) -> None:
    model = DummyModel(
        scripted_messages=[
            assistant_action_message(name="search", arguments='{"query": "fed rates"}', cost=0.11),
        ]
    )
    env = DummyEnvironment(outputs=[{"output": "ok", "search_cost": 0.25}])
    ctx = _RecordingContext()
    kwargs = _agent_kwargs(tmp_path)
    kwargs["step_limit"] = 1
    agent = DefaultForecastAgent(model=model, env=env, context_manager=ctx, **kwargs)

    result = agent.run_sync(title="Q")

    assert result["exit_status"] == "LimitsExceeded"
    assert agent.model_cost == pytest.approx(0.11)
    assert agent.search_cost == pytest.approx(0.25)
    assert agent.n_searches == 1
    assert ctx.queries == ["fed rates"]


def test_default_agent_tracks_cache_tokens(
    tmp_path: Path,
) -> None:
    """Cache tokens are accumulated across steps and appear in serialize_info."""
    import time

    model = DummyModel(
        scripted_messages=[
            {
                "role": "assistant",
                "content": "",
                "extra": {
                    "actions": [
                        {"name": "search", "arguments": '{"query": "q1"}', "tool_call_id": "t1"}
                    ],
                    "cost": 0.01,
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "cached_tokens": 200,
                    "cache_creation_tokens": 800,
                    "timestamp": time.time(),
                },
            },
            {
                "role": "assistant",
                "content": "",
                "extra": {
                    "actions": [
                        {"name": "search", "arguments": '{"query": "q2"}', "tool_call_id": "t2"}
                    ],
                    "cost": 0.01,
                    "prompt_tokens": 1500,
                    "completion_tokens": 60,
                    "cached_tokens": 900,
                    "cache_creation_tokens": 0,
                    "timestamp": time.time(),
                },
            },
        ]
    )
    env = DummyEnvironment()
    kwargs = _agent_kwargs(tmp_path)
    kwargs["step_limit"] = 2
    agent = DefaultForecastAgent(model=model, env=env, **kwargs)
    agent.run_sync(title="Q")

    # Latest per-call values should be from the second call
    assert agent.cached_tokens == 900
    assert agent.cache_creation_tokens == 0

    # Cumulative totals
    assert agent.total_prompt_tokens == 2500  # 1000 + 1500
    assert agent.total_cached_tokens == 1100  # 200 + 900

    # Serialization
    info = agent.serialize_info()
    tu = info["token_usage"]
    assert tu["total_prompt_tokens"] == 2500
    assert tu["total_cached_tokens"] == 1100
    assert tu["cache_hit_rate"] == pytest.approx(1100 / 2500)


def test_default_agent_cache_tokens_none_when_unsupported(
    tmp_path: Path,
) -> None:
    """When model doesn't provide cache tokens, values remain None/zero."""
    import time

    model = DummyModel(
        scripted_messages=[
            {
                "role": "assistant",
                "content": "",
                "extra": {
                    "actions": [
                        {"name": "search", "arguments": '{"query": "q"}', "tool_call_id": "t1"}
                    ],
                    "cost": 0.01,
                    "prompt_tokens": 1000,
                    "completion_tokens": 50,
                    "timestamp": time.time(),
                },
            },
        ]
    )
    env = DummyEnvironment()
    kwargs = _agent_kwargs(tmp_path)
    kwargs["step_limit"] = 1
    agent = DefaultForecastAgent(model=model, env=env, **kwargs)
    agent.run_sync(title="Q")

    assert agent.cached_tokens is None
    assert agent.total_cached_tokens == 0

    info = agent.serialize_info()
    assert info["token_usage"]["cache_hit_rate"] is None
