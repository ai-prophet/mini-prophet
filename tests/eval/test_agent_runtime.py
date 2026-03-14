from __future__ import annotations

import threading

import pytest
from conftest import DummyEnvironment, DummyModel

from miniprophet.agent.default import DefaultForecastAgent
from miniprophet.eval.agent_runtime import EvalBatchAgentWrapper, RateLimitCoordinator
from miniprophet.exceptions import BatchRunTimeoutError


def test_cancel_event_stops_agent_step() -> None:
    """DefaultForecastAgent.step() should raise when cancel_event is set."""
    cancel = threading.Event()
    cancel.set()

    agent = DefaultForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        cancel_event=cancel,
        system_template="sys",
        instance_template="inst",
    )

    with pytest.raises(BatchRunTimeoutError, match="cancelled"):
        agent.step()


def test_wrapper_pre_run_guard_checks_cancel_event() -> None:
    """EvalBatchAgentWrapper should raise on run() when cancel_event is set."""
    cancel = threading.Event()
    cancel.set()

    agent = DefaultForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        system_template="sys",
        instance_template="inst",
    )
    wrapper = EvalBatchAgentWrapper(
        agent=agent,
        task_id="run-x",
        cancel_event=cancel,
    )

    with pytest.raises(BatchRunTimeoutError, match="timed out"):
        wrapper.run(title="test", outcomes=["Yes", "No"])


def test_rate_limit_coordinator_wait_cancelled() -> None:
    coordinator = RateLimitCoordinator(backoff_seconds=0.2)
    coordinator.signal_rate_limit(backoff_seconds=0.2)
    cancel = threading.Event()
    cancel.set()

    assert coordinator.wait_if_paused(cancel_event=cancel) is False
