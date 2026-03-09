"""Batch/eval runtime wrappers for default and custom agents."""

from __future__ import annotations

import logging
import threading
from typing import Any

from miniprophet import ContextManager, Environment, Model
from miniprophet.agent.default import AgentConfig, DefaultForecastAgent
from miniprophet.eval.progress import EvalProgressManager
from miniprophet.exceptions import BatchRunTimeoutError

logger = logging.getLogger("miniprophet.agent.eval")


class RateLimitCoordinator:
    """Shared pause/resume gate for all eval workers."""

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


class BatchForecastAgent(DefaultForecastAgent):
    """Default agent variant with eval-safe pause/cancel and progress hooks."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        context_manager: ContextManager | None = None,
        config_class: type = AgentConfig,
        task_id: str = "",
        coordinator: RateLimitCoordinator | None = None,
        progress_manager: EvalProgressManager | None = None,
        cancel_event: threading.Event | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            env=env,
            context_manager=context_manager,
            config_class=config_class,
            **kwargs,
        )
        self.task_id = task_id
        self._coordinator = coordinator
        self._progress = progress_manager
        self._cancel_event = cancel_event
        self._prev_cost = 0.0

    def _check_cancelled(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise BatchRunTimeoutError("Eval run timed out and was cancelled.")

    def step(self) -> list[dict]:
        self._check_cancelled()

        if self._coordinator is not None:
            resumed = self._coordinator.wait_if_paused(cancel_event=self._cancel_event)
            if not resumed:
                raise BatchRunTimeoutError("Eval run timed out and was cancelled.")

        res = super().step()

        self._check_cancelled()

        cost_delta = self.total_cost - self._prev_cost
        self._prev_cost = self.total_cost

        if self._progress is not None:
            self._progress.update_run_status(
                self.task_id,
                f"Step {self.n_calls + 1} (${self.total_cost:.2f}/{self.config.cost_limit:.2f})",
                cost_delta=cost_delta,
            )
        return res


class EvalBatchAgentWrapper:
    """Wrap arbitrary agents so they can run in eval worker orchestration."""

    def __init__(
        self,
        *,
        agent: Any,
        task_id: str,
        coordinator: RateLimitCoordinator | None,
        progress_manager: EvalProgressManager | None,
        cancel_event: threading.Event | None,
    ) -> None:
        self._agent = agent
        self.task_id = task_id
        self._coordinator = coordinator
        self._progress = progress_manager
        self._cancel_event = cancel_event
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

    def _check_cancelled(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise BatchRunTimeoutError("Eval run timed out and was cancelled.")

    def _pre_step_guard(self) -> None:
        self._check_cancelled()

        if self._coordinator is not None:
            resumed = self._coordinator.wait_if_paused(cancel_event=self._cancel_event)
            if not resumed:
                raise BatchRunTimeoutError("Eval run timed out and was cancelled.")

    def _patch_step(self) -> tuple[Any | None, Any | None]:
        original_step = getattr(self._agent, "step", None)
        if original_step is None or not callable(original_step):
            return None, None

        def _wrapped_step(*args, **kwargs):
            self._pre_step_guard()
            res = original_step(*args, **kwargs)
            self._pre_step_guard()
            if self._progress is not None:
                cost_delta = self.total_cost - self._prev_cost
                self._prev_cost = self.total_cost
                step_idx = int(getattr(self._agent, "n_calls", 0) or 0) + 1
                self._progress.update_run_status(
                    self.task_id,
                    f"Step {step_idx} (${self.total_cost:.2f})",
                    cost_delta=cost_delta,
                )
            return res

        setattr(self._agent, "step", _wrapped_step)
        return original_step, _wrapped_step

    def _restore_step(self, original_step: Any | None, wrapped_step: Any | None) -> None:
        if original_step is None or wrapped_step is None:
            return
        current = getattr(self._agent, "step", None)
        if current is wrapped_step:
            setattr(self._agent, "step", original_step)

    def run(
        self,
        *,
        title: str,
        outcomes: list[str],
        ground_truth: dict[str, int] | None = None,
        **runtime_kwargs,
    ) -> Any:
        self._pre_step_guard()
        original_step, wrapped_step = self._patch_step()
        try:
            return self._agent.run(
                title=title,
                outcomes=outcomes,
                ground_truth=ground_truth,
                **runtime_kwargs,
            )
        finally:
            self._restore_step(original_step, wrapped_step)

    def save(self, path, *extra_dicts):
        save = getattr(self._agent, "save", None)
        if callable(save):
            return save(path, *extra_dicts)
        return {}
