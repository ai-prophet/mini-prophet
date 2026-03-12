"""Tests for the grace period feature of DefaultForecastAgent."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import DummyEnvironment, DummyModel

from miniprophet.agent.default import DefaultForecastAgent
from miniprophet.exceptions import Submitted


class _SubmitEnv(DummyEnvironment):
    """Environment that handles submit by raising Submitted."""

    def execute(self, action: dict, **kwargs) -> dict:
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
        return super().execute(action, **kwargs)


TEST_GRACE_PROMPT = "Submit now using the submit tool."


def _grace_kwargs(tmp_path: Path, **overrides) -> dict:
    defaults = {
        "system_template": "sys: {title}",
        "instance_template": "inst: {title} {outcomes_formatted} {current_time}",
        "step_limit": 2,
        "cost_limit": 99.0,
        "search_limit": 9,
        "max_outcomes": 10,
        "output_path": tmp_path,
        "enable_grace_period": True,
        "grace_period_extra_turns": 3,
        "grace_period_prompt": TEST_GRACE_PROMPT,
    }
    defaults.update(overrides)
    return defaults


def test_grace_period_allows_submit_after_step_limit(
    assistant_action_message, tmp_path: Path
) -> None:
    """Agent submits successfully during grace period after step limit is reached."""
    model = DummyModel(
        scripted_messages=[
            # Step 1: a normal search action
            assistant_action_message(name="search", arguments='{"query": "test"}'),
            # Step 2: another search (consumes step_limit=2)
            assistant_action_message(name="search", arguments='{"query": "test2"}'),
            # Grace period turn 1: agent submits
            assistant_action_message(
                name="submit", arguments='{"probabilities": {"Yes": 0.7, "No": 0.3}}'
            ),
        ]
    )
    env = _SubmitEnv()
    agent = DefaultForecastAgent(model=model, env=env, **_grace_kwargs(tmp_path))

    result = agent.run(title="Q", outcomes=["Yes", "No"])

    assert result["exit_status"] == "submitted"
    assert result["submission"] == {"Yes": 0.7, "No": 0.3}
    assert agent._in_grace_period is True
    assert agent._grace_period_turns == 1


def test_grace_period_rejects_non_submit_tools(
    assistant_action_message, tmp_path: Path
) -> None:
    """Non-submit tool calls during grace period are rejected with grace period prompt."""
    model = DummyModel(
        scripted_messages=[
            # Step 1: normal action
            assistant_action_message(name="search", arguments='{"query": "test"}'),
            # Step 2: normal action (fills step_limit=2)
            assistant_action_message(name="search", arguments='{"query": "test2"}'),
            # Grace turn 1: tries search (rejected)
            assistant_action_message(name="search", arguments='{"query": "test3"}'),
            # Grace turn 2: submits
            assistant_action_message(
                name="submit", arguments='{"probabilities": {"Yes": 0.7, "No": 0.3}}'
            ),
        ]
    )
    env = _SubmitEnv()
    agent = DefaultForecastAgent(model=model, env=env, **_grace_kwargs(tmp_path))

    result = agent.run(title="Q", outcomes=["Yes", "No"])

    assert result["exit_status"] == "submitted"
    assert agent._grace_period_turns == 2


def test_grace_period_exhausted_after_extra_turns(
    assistant_action_message, tmp_path: Path
) -> None:
    """Agent fails after exhausting all grace period turns without submitting."""
    model = DummyModel(
        scripted_messages=[
            # Step 1: normal action
            assistant_action_message(name="search", arguments='{"query": "test"}'),
            # Step 2: fills step_limit=2
            assistant_action_message(name="search", arguments='{"query": "test2"}'),
            # Grace turns 1-3: all non-submit (grace_period_extra_turns=3)
            assistant_action_message(name="search", arguments='{"query": "g1"}'),
            assistant_action_message(name="search", arguments='{"query": "g2"}'),
            assistant_action_message(name="search", arguments='{"query": "g3"}'),
        ]
    )
    env = _SubmitEnv()
    agent = DefaultForecastAgent(model=model, env=env, **_grace_kwargs(tmp_path))

    result = agent.run(title="Q", outcomes=["Yes", "No"])

    assert result["exit_status"] == "LimitsExceeded"
    assert agent._grace_period_turns == 3


def test_grace_period_disabled_fails_immediately(
    assistant_action_message, tmp_path: Path
) -> None:
    """Without grace period enabled, step limit causes immediate failure."""
    model = DummyModel(
        scripted_messages=[
            assistant_action_message(name="search", arguments='{"query": "test"}'),
            assistant_action_message(name="search", arguments='{"query": "test2"}'),
        ]
    )
    env = DummyEnvironment()
    kwargs = _grace_kwargs(tmp_path, enable_grace_period=False)
    agent = DefaultForecastAgent(model=model, env=env, **kwargs)

    result = agent.run(title="Q", outcomes=["Yes", "No"])

    assert result["exit_status"] == "LimitsExceeded"
    assert agent._in_grace_period is False


def test_grace_period_prompt_injected_as_user_message(
    assistant_action_message, tmp_path: Path
) -> None:
    """Grace period prompt is appended as a user message before querying the model."""
    model = DummyModel(
        scripted_messages=[
            assistant_action_message(name="search", arguments='{"query": "test"}'),
            assistant_action_message(name="search", arguments='{"query": "test2"}'),
            # Grace turn: submit
            assistant_action_message(
                name="submit", arguments='{"probabilities": {"Yes": 0.7, "No": 0.3}}'
            ),
        ]
    )
    env = _SubmitEnv()
    agent = DefaultForecastAgent(model=model, env=env, **_grace_kwargs(tmp_path))

    agent.run(title="Q", outcomes=["Yes", "No"])

    # Find the grace period prompt in messages
    grace_messages = [
        m for m in agent.messages if m.get("content") == TEST_GRACE_PROMPT and m.get("role") == "user"
    ]
    assert len(grace_messages) >= 1


def test_grace_period_custom_prompt(
    assistant_action_message, tmp_path: Path
) -> None:
    """Custom grace period prompt is used when configured."""
    custom_prompt = "Custom: submit now!"
    model = DummyModel(
        scripted_messages=[
            assistant_action_message(name="search", arguments='{"query": "test"}'),
            assistant_action_message(name="search", arguments='{"query": "test2"}'),
            assistant_action_message(
                name="submit", arguments='{"probabilities": {"Yes": 0.7, "No": 0.3}}'
            ),
        ]
    )
    env = _SubmitEnv()
    kwargs = _grace_kwargs(tmp_path, grace_period_prompt=custom_prompt)
    agent = DefaultForecastAgent(model=model, env=env, **kwargs)

    agent.run(title="Q", outcomes=["Yes", "No"])

    custom_messages = [
        m for m in agent.messages if m.get("content") == custom_prompt and m.get("role") == "user"
    ]
    assert len(custom_messages) >= 1


def test_grace_period_keeps_all_tool_schemas(
    assistant_action_message, tmp_path: Path
) -> None:
    """During grace period, all tool schemas are still provided for KV cache friendliness."""
    tools_seen: list[list[dict]] = []

    class _RecordingModel(DummyModel):
        def query(self, messages, tools):
            tools_seen.append(tools)
            return super().query(messages, tools)

    model = _RecordingModel(
        scripted_messages=[
            assistant_action_message(name="search", arguments='{"query": "test"}'),
            assistant_action_message(name="search", arguments='{"query": "test2"}'),
            assistant_action_message(
                name="submit", arguments='{"probabilities": {"Yes": 0.7, "No": 0.3}}'
            ),
        ]
    )

    env = _SubmitEnv()
    env._tools["submit"] = {
        "type": "function",
        "function": {"name": "submit", "parameters": {}},
    }
    agent = DefaultForecastAgent(model=model, env=env, **_grace_kwargs(tmp_path))

    agent.run(title="Q", outcomes=["Yes", "No"])

    # All queries should receive the same full set of tool schemas
    assert tools_seen[-1] == tools_seen[0]
