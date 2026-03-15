from __future__ import annotations

import asyncio

from conftest import DummyEnvironment, DummyModel

from miniprophet.agent.default import DefaultForecastAgent
from miniprophet.eval.agent_runtime import EvalBatchAgentWrapper, RateLimitCoordinator


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

        # Should be paused now — wait with timeout to prove it resumes
        await asyncio.wait_for(coordinator.wait_if_paused(), timeout=1.0)

    asyncio.run(_test())
