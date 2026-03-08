from __future__ import annotations

import threading

import pytest
from conftest import DummyEnvironment, DummyModel

from miniprophet.eval.agent_runtime import BatchForecastAgent, RateLimitCoordinator
from miniprophet.exceptions import BatchRunTimeoutError


class _Progress:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def update_run_status(self, task_id: str, message: str, cost_delta: float = 0.0) -> None:
        self.calls.append((task_id, message, cost_delta))


def test_batch_forecast_agent_step_updates_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_super_step(self):
        self.model_cost = 1.5
        return [{"role": "tool", "content": "ok"}]

    monkeypatch.setattr(
        "miniprophet.eval.agent_runtime.DefaultForecastAgent.step", _fake_super_step
    )

    progress = _Progress()
    agent = BatchForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        task_id="run-x",
        progress_manager=progress,
        system_template="sys",
        instance_template="inst",
    )

    out = agent.step()

    assert out[0]["content"] == "ok"
    assert progress.calls[0][0] == "run-x"
    assert progress.calls[0][2] == pytest.approx(1.5)


def test_batch_forecast_agent_step_respects_cancel_event() -> None:
    cancel = threading.Event()
    cancel.set()

    agent = BatchForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        cancel_event=cancel,
        system_template="sys",
        instance_template="inst",
    )

    with pytest.raises(BatchRunTimeoutError, match="timed out"):
        agent.step()


def test_rate_limit_coordinator_wait_cancelled() -> None:
    coordinator = RateLimitCoordinator(backoff_seconds=0.2)
    coordinator.signal_rate_limit(backoff_seconds=0.2)
    cancel = threading.Event()
    cancel.set()

    assert coordinator.wait_if_paused(cancel_event=cancel) is False
