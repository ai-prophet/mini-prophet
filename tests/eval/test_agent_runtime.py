from __future__ import annotations

import threading

import pytest
from conftest import DummyEnvironment, DummyModel

from miniprophet.agent.default import DefaultForecastAgent
from miniprophet.eval.agent_runtime import EvalBatchAgentWrapper, RateLimitCoordinator
from miniprophet.exceptions import BatchRunTimeoutError


class _Progress:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def update_run_status(self, task_id: str, message: str, cost_delta: float = 0.0) -> None:
        self.calls.append((task_id, message, cost_delta))


def test_wrapper_around_default_agent_step_updates_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_step(self):
        self.model_cost = 1.5
        return [{"role": "tool", "content": "ok"}]

    monkeypatch.setattr("miniprophet.agent.default.DefaultForecastAgent.step", _fake_step)

    progress = _Progress()
    raw_agent = DefaultForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        system_template="sys",
        instance_template="inst",
    )
    wrapper = EvalBatchAgentWrapper(
        agent=raw_agent,
        task_id="run-x",
        progress_manager=progress,
    )

    # Patch step via the wrapper's run-time mechanism
    wrapper._patch_step()
    out = raw_agent.step()

    assert out[0]["content"] == "ok"
    assert progress.calls[0][0] == "run-x"
    assert progress.calls[0][2] == pytest.approx(1.5)
    # Should include cost_limit in message since DefaultForecastAgent has config.cost_limit
    assert "/" in progress.calls[0][1]


def test_wrapper_step_respects_cancel_event() -> None:
    cancel = threading.Event()
    cancel.set()

    raw_agent = DefaultForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        system_template="sys",
        instance_template="inst",
    )
    wrapper = EvalBatchAgentWrapper(
        agent=raw_agent,
        task_id="run-x",
        cancel_event=cancel,
    )
    wrapper._patch_step()

    with pytest.raises(BatchRunTimeoutError, match="timed out"):
        raw_agent.step()


def test_rate_limit_coordinator_wait_cancelled() -> None:
    coordinator = RateLimitCoordinator(backoff_seconds=0.2)
    coordinator.signal_rate_limit(backoff_seconds=0.2)
    cancel = threading.Event()
    cancel.set()

    assert coordinator.wait_if_paused(cancel_event=cancel) is False
