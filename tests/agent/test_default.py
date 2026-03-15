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
        "instance_template": "inst: {title} {outcomes_formatted} {current_time}",
        "step_limit": 2,
        "cost_limit": 99.0,
        "search_limit": 9,
        "max_outcomes": 10,
        "output_path": tmp_path,
    }


def test_default_agent_run_submitted_and_evaluated(
    assistant_action_message,
    tmp_path: Path,
) -> None:
    model = DummyModel(
        scripted_messages=[
            assistant_action_message(
                name="submit", arguments='{"probabilities": {"Yes": 0.7, "No": 0.3}}'
            )
        ]
    )
    env = _SubmitEnv()
    agent = DefaultForecastAgent(model=model, env=env, **_agent_kwargs(tmp_path))

    result = agent.run_sync(
        title="Will X happen?",
        outcomes=["Yes", "No"],
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

    result = agent.run_sync(title="Q", outcomes=["A", "B"])

    assert result["exit_status"] == "LimitsExceeded"
    assert agent.model_cost == pytest.approx(0.11)
    assert agent.search_cost == pytest.approx(0.25)
    assert agent.n_searches == 1
    assert ctx.queries == ["fed rates"]


def test_default_agent_rejects_invalid_outcome_count(tmp_path: Path) -> None:
    agent = DefaultForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        **{
            **_agent_kwargs(tmp_path),
            "max_outcomes": 1,
        },
    )

    with pytest.raises(ValueError, match="Too many outcomes"):
        agent.run_sync(title="Q", outcomes=["A", "B"])
