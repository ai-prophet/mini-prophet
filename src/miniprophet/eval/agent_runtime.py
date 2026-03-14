"""Batch/eval runtime wrappers for default and custom agents."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger("miniprophet.agent.eval")


class RateLimitCoordinator:
    """Shared pause/resume gate for all eval workers (asyncio-based)."""

    def __init__(self, backoff_seconds: float = 60) -> None:
        self._resume = asyncio.Event()
        self._resume.set()
        self._lock = asyncio.Lock()
        self._backoff = backoff_seconds

    async def wait_if_paused(self) -> None:
        await self._resume.wait()

    async def signal_rate_limit(self, backoff_seconds: float | None = None) -> None:
        secs = backoff_seconds if backoff_seconds is not None else self._backoff
        async with self._lock:
            if self._resume.is_set():
                logger.warning("Rate limit detected -- pausing all workers for %.0fs", secs)
                self._resume.clear()
                asyncio.get_running_loop().call_later(secs, self._resume.set)


# Keep the threading-based coordinator for backward compat with sync code paths
class ThreadedRateLimitCoordinator:
    """Shared pause/resume gate for all eval workers (threading-based, legacy)."""

    def __init__(self, backoff_seconds: float = 60) -> None:
        self._resume = threading.Event()
        self._resume.set()
        self._lock = threading.Lock()
        self._backoff = backoff_seconds

    def wait_if_paused(self, cancel_event: threading.Event | None = None) -> bool:
        if cancel_event is not None and cancel_event.is_set():
            return False
        while not self._resume.wait(timeout=1.0):
            if cancel_event is not None and cancel_event.is_set():
                return False
        return True

    def signal_rate_limit(self, backoff_seconds: float | None = None) -> None:
        secs = backoff_seconds if backoff_seconds is not None else self._backoff
        with self._lock:
            if self._resume.is_set():
                logger.warning("Rate limit detected -- pausing all workers for %.0fs", secs)
                self._resume.clear()
                timer = threading.Timer(secs, self._resume.set)
                timer.daemon = True
                timer.start()


class EvalBatchAgentWrapper:
    """Wrap arbitrary agents so they can run in eval worker orchestration.

    Timeout cancellation is handled by ``DefaultForecastAgent`` directly
    via the ``cancel_event`` parameter (passed through by
    ``EvalAgentFactory``).  This wrapper handles pre-run guards
    (rate-limit coordination) and cost tracking.
    """

    def __init__(
        self,
        *,
        agent: Any,
        task_id: str,
        coordinator: RateLimitCoordinator | None = None,
        progress_manager: Any | None = None,
    ) -> None:
        self._agent = agent
        self.task_id = task_id
        self._coordinator = coordinator
        self._progress = progress_manager
        self._cancel_event = cancel_event

    @property
    def model_cost(self) -> float:
        return float(getattr(self._agent, "model_cost", 0.0) or 0.0)

    @property
    def search_cost(self) -> float:
        return float(getattr(self._agent, "search_cost", 0.0) or 0.0)

    @property
    def total_cost(self) -> float:
        val = getattr(self._agent, "total_cost", None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
        return self.model_cost + self.search_cost

    def _pre_run_guard(self) -> None:
        """Check cancel event and wait for rate-limit coordinator before starting."""
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise BatchRunTimeoutError("Eval run timed out and was cancelled.")

        if self._coordinator is not None:
            await self._coordinator.wait_if_paused()

    def run(
        self,
        *,
        title: str,
        outcomes: list[str],
        ground_truth: dict[str, int] | None = None,
        **runtime_kwargs,
    ) -> Any:
        self._pre_run_guard()
        return self._agent.run(
            title=title,
            outcomes=outcomes,
            ground_truth=ground_truth,
            **runtime_kwargs,
        )

    def save(self, path, *extra_dicts):
        save = getattr(self._agent, "save", None)
        if callable(save):
            return save(path, *extra_dicts)
        return {}
