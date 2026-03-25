from __future__ import annotations

import asyncio

from conftest import DummyEnvironment, DummyModel

from miniprophet.agent.default import DefaultForecastAgent
from miniprophet.eval.agent_runtime import EvalBatchAgentWrapper, RateLimitCoordinator


class _Progress:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def update_run_status(self, task_id: str, message: str, cost_delta: float = 0.0) -> None:
        self.calls.append((task_id, message, cost_delta))


def test_wrapper_run_updates_progress() -> None:
    """EvalBatchAgentWrapper.run() tracks progress via hooked step."""
    progress = _Progress()
    raw_agent = DefaultForecastAgent(
        model=DummyModel(),
        env=DummyEnvironment(),
        system_template="sys",
        instance_template="inst: {title} {current_time}",
        step_limit=1,
    )
    wrapper = EvalBatchAgentWrapper(
        agent=raw_agent,
        task_id="run-x",
        progress_manager=progress,
    )

    asyncio.run(
        wrapper.run(
            title="Q",
        )
    )

    # At least one progress update should have been recorded
    assert len(progress.calls) >= 1
    assert progress.calls[0][0] == "run-x"
    # Should include cost_limit in message since DefaultForecastAgent has config.cost_limit
    assert "/" in progress.calls[0][1]


def test_rate_limit_coordinator_async() -> None:
    """RateLimitCoordinator async wait/signal behavior."""

    async def _test():
        coordinator = RateLimitCoordinator(backoff_seconds=0.1)
        # Initially not paused
        await coordinator.wait_if_paused()

        # Signal rate limit
        await coordinator.signal_rate_limit(backoff_seconds=0.1)

        # Should be paused now — wait with timeout to prove it resumes
        await asyncio.wait_for(coordinator.wait_if_paused(), timeout=1.0)

    asyncio.run(_test())
