"""Batch/eval runtime wrappers for default and custom agents."""

from __future__ import annotations

import asyncio
import logging
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


class EvalBatchAgentWrapper:
    """Wrap arbitrary agents so they can run in eval worker orchestration."""

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
        self._prev_cost = 0.0

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

    async def _pre_step_guard(self) -> None:
        if self._coordinator is not None:
            await self._coordinator.wait_if_paused()

    def _update_progress(self) -> None:
        if self._progress is not None:
            cost_delta = self.total_cost - self._prev_cost
            self._prev_cost = self.total_cost
            step_idx = int(getattr(self._agent, "n_calls", 0) or 0) + 1
            cost_limit = getattr(getattr(self._agent, "config", None), "cost_limit", None)
            if cost_limit is not None:
                msg = f"Step {step_idx} (${self.total_cost:.2f}/${cost_limit:.2f})"
            else:
                msg = f"Step {step_idx} (${self.total_cost:.2f})"
            self._progress.update_run_status(
                self.task_id,
                msg,
                cost_delta=cost_delta,
            )

    async def run(
        self,
        *,
        title: str,
        ground_truth: dict[str, int] | None = None,
        **runtime_kwargs,
    ) -> Any:
        await self._pre_step_guard()
        original_step = self._agent.step

        async def _hooked_step(*args, **kwargs):
            await self._pre_step_guard()
            res = await original_step(*args, **kwargs)
            self._update_progress()
            return res

        self._agent.step = _hooked_step
        try:
            return await self._agent.run(
                title=title,
                ground_truth=ground_truth,
                **runtime_kwargs,
            )
        finally:
            self._agent.step = original_step

    def run_sync(
        self,
        *,
        title: str,
        ground_truth: dict[str, int] | None = None,
        **runtime_kwargs,
    ) -> Any:
        return asyncio.run(
            self.run(
                title=title,
                ground_truth=ground_truth,
                **runtime_kwargs,
            )
        )

    def save(self, path, *extra_dicts):
        save = getattr(self._agent, "save", None)
        if callable(save):
            return save(path, *extra_dicts)
        return {}
