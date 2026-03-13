"""Tests for CliForecastAgent interactive interrupt (Ctrl+C pause & message injection)."""

from __future__ import annotations

import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import DummyEnvironment, DummyModel

from miniprophet.agent.cli_agent import CliForecastAgent
from miniprophet.exceptions import Submitted


class _CliDummyEnvironment(DummyEnvironment):
    """DummyEnvironment with _tools cleared so on_observation doesn't call .display()."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tools = {}  # No tool objects — avoids .display() calls in CLI hook


class _SubmitEnv(_CliDummyEnvironment):
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


def _agent_kwargs(tmp_path: Path) -> dict:
    return {
        "system_template": "sys: {title}",
        "instance_template": "inst: {title} {outcomes_formatted} {current_time}",
        "step_limit": 10,
        "cost_limit": 99.0,
        "search_limit": 9,
        "max_outcomes": 10,
        "output_path": tmp_path,
        "enable_interrupt": True,
    }


def _make_agent(
    tmp_path: Path,
    scripted_messages: list[dict],
    env: DummyEnvironment | None = None,
) -> CliForecastAgent:
    model = DummyModel(scripted_messages=scripted_messages)
    if env is None:
        env = _SubmitEnv()
    return CliForecastAgent(model=model, env=env, **_agent_kwargs(tmp_path))


# ---- Test: interrupt with tool action yields A -> B -> C ----


@patch("miniprophet.agent.cli_agent.Prompt.ask", return_value="Focus on Europe")
@patch("miniprophet.agent.cli_agent.console")
def test_interrupt_after_tool_action_injects_user_message(
    mock_console,
    mock_prompt,
    assistant_action_message,
    tmp_path: Path,
) -> None:
    """When interrupt is requested and model returned a tool call,
    execute the tool (A -> B), then inject user message (C)."""
    agent = _make_agent(
        tmp_path,
        scripted_messages=[
            assistant_action_message(name="search", arguments='{"query": "test"}'),
            assistant_action_message(
                name="submit",
                arguments='{"probabilities": {"Yes": 0.7, "No": 0.3}}',
            ),
        ],
    )

    # Set the interrupt flag before step 1
    original_step = agent.step

    def step_with_interrupt():
        # Set flag before the first step only
        if agent.n_calls == 0:
            agent._interrupt_requested = True
        return original_step()

    agent.step = step_with_interrupt

    result = agent.run(title="Q?", outcomes=["Yes", "No"])

    # The user message should be in the conversation
    user_interrupt_msgs = [m for m in agent.messages if m.get("extra", {}).get("is_user_interrupt")]
    assert len(user_interrupt_msgs) == 1
    assert user_interrupt_msgs[0]["content"] == "Focus on Europe"
    assert user_interrupt_msgs[0]["role"] == "user"

    # Agent should have completed (submitted)
    assert result["exit_status"] == "submitted"


# ---- Test: interrupt with no actions yields A -> C ----


@patch("miniprophet.agent.cli_agent.Prompt.ask", return_value="Try a different approach")
@patch("miniprophet.agent.cli_agent.console")
def test_interrupt_no_action_injects_user_message_directly(
    mock_console,
    mock_prompt,
    tmp_path: Path,
) -> None:
    """When model returns text-only (no tool calls) and interrupt is set,
    inject user message directly after assistant message (A -> C)."""
    # First message: text-only (no actions). Second: submit.
    text_only_msg = {
        "role": "assistant",
        "content": "Let me think about this...",
        "extra": {"actions": [], "cost": 0.01, "timestamp": 0.0},
    }

    agent = _make_agent(
        tmp_path,
        scripted_messages=[
            text_only_msg,
            {
                "role": "assistant",
                "content": "",
                "extra": {
                    "actions": [
                        {
                            "name": "submit",
                            "arguments": '{"probabilities": {"Yes": 0.5, "No": 0.5}}',
                            "tool_call_id": "tc_1",
                        }
                    ],
                    "cost": 0.0,
                    "timestamp": 0.0,
                },
            },
        ],
    )

    original_step = agent.step

    def step_with_interrupt():
        if agent.n_calls == 0:
            agent._interrupt_requested = True
        return original_step()

    agent.step = step_with_interrupt

    agent.run(title="Q?", outcomes=["Yes", "No"])

    user_interrupt_msgs = [m for m in agent.messages if m.get("extra", {}).get("is_user_interrupt")]
    assert len(user_interrupt_msgs) == 1
    assert user_interrupt_msgs[0]["content"] == "Try a different approach"

    # Verify ordering: assistant text-only, then user interrupt, then next steps
    roles_after_init = [
        m["role"] for m in agent.messages[2:] if not m.get("extra", {}).get("is_board_state")
    ]
    # Should have: assistant (text-only), user (interrupt), assistant (submit), exit
    assert "user" in roles_after_init


# ---- Test: empty input does not inject a message ----


@patch("miniprophet.agent.cli_agent.Prompt.ask", return_value="")
@patch("miniprophet.agent.cli_agent.console")
def test_empty_input_does_not_inject_message(
    mock_console,
    mock_prompt,
    assistant_action_message,
    tmp_path: Path,
) -> None:
    agent = _make_agent(
        tmp_path,
        scripted_messages=[
            assistant_action_message(name="search", arguments='{"query": "test"}'),
            assistant_action_message(
                name="submit",
                arguments='{"probabilities": {"Yes": 0.6, "No": 0.4}}',
            ),
        ],
    )

    original_step = agent.step

    def step_with_interrupt():
        if agent.n_calls == 0:
            agent._interrupt_requested = True
        return original_step()

    agent.step = step_with_interrupt

    result = agent.run(title="Q?", outcomes=["Yes", "No"])

    user_interrupt_msgs = [m for m in agent.messages if m.get("extra", {}).get("is_user_interrupt")]
    assert len(user_interrupt_msgs) == 0
    assert result["exit_status"] == "submitted"


# ---- Test: double Ctrl+C raises KeyboardInterrupt ----


def test_double_ctrl_c_raises_keyboard_interrupt(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, scripted_messages=[])

    # First call sets the flag
    agent._handle_sigint(signal.SIGINT, None)
    assert agent._interrupt_requested is True

    # Second call should raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        agent._handle_sigint(signal.SIGINT, None)


# ---- Test: signal handler installed and restored around run() ----


@patch("miniprophet.agent.cli_agent.Prompt.ask", return_value="")
@patch("miniprophet.agent.cli_agent.console")
def test_signal_handler_restored_after_run(
    mock_console,
    mock_prompt,
    assistant_action_message,
    tmp_path: Path,
) -> None:
    agent = _make_agent(
        tmp_path,
        scripted_messages=[
            assistant_action_message(
                name="submit",
                arguments='{"probabilities": {"Yes": 0.5, "No": 0.5}}',
            ),
        ],
    )

    original_handler = signal.getsignal(signal.SIGINT)

    agent.run(title="Q?", outcomes=["Yes", "No"])

    # After run, the original handler should be restored
    restored_handler = signal.getsignal(signal.SIGINT)
    assert restored_handler is original_handler


# ---- Test: signal handler restored even on exception ----


@patch("miniprophet.agent.cli_agent.console")
def test_signal_handler_restored_on_exception(
    mock_console,
    tmp_path: Path,
) -> None:
    # Model that raises an exception
    model = DummyModel(scripted_messages=[])
    model.query = MagicMock(side_effect=RuntimeError("boom"))
    env = DummyEnvironment()
    agent = CliForecastAgent(model=model, env=env, **_agent_kwargs(tmp_path))

    original_handler = signal.getsignal(signal.SIGINT)

    with pytest.raises(RuntimeError, match="boom"):
        agent.run(title="Q?", outcomes=["Yes", "No"])

    restored_handler = signal.getsignal(signal.SIGINT)
    assert restored_handler is original_handler


# ---- Test: interactive disabled skips signal handler ----


@patch("miniprophet.agent.cli_agent.console")
def test_interactive_disabled_no_signal_handler(
    mock_console,
    assistant_action_message,
    tmp_path: Path,
) -> None:
    kwargs = _agent_kwargs(tmp_path)
    kwargs["enable_interrupt"] = False
    model = DummyModel(
        scripted_messages=[
            assistant_action_message(
                name="submit",
                arguments='{"probabilities": {"Yes": 0.5, "No": 0.5}}',
            ),
        ]
    )
    env = _SubmitEnv()
    agent = CliForecastAgent(model=model, env=env, **kwargs)

    original_handler = signal.getsignal(signal.SIGINT)
    agent.run(title="Q?", outcomes=["Yes", "No"])
    # Handler should never have changed
    assert signal.getsignal(signal.SIGINT) is original_handler
