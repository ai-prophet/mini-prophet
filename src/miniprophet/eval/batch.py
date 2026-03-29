"""Public batch forecasting API."""

from __future__ import annotations

import asyncio
import copy
import logging
from pathlib import Path
from typing import Any

from miniprophet.eval.agent_factory import EvalAgentFactory
from miniprophet.eval.agent_runtime import RateLimitCoordinator
from miniprophet.eval.types import BatchProgressCallback, ForecastProblem, ForecastResult
from miniprophet.exceptions import (
    BatchFatalError,
    SearchAuthError,
    SearchRateLimitError,
)

logger = logging.getLogger("miniprophet.eval.batch")

MAX_RETRIES = 3
RETRYABLE_EXCEPTIONS = (SearchRateLimitError,)


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True
    exc_str = str(exc).lower()
    return "429" in exc_str or "rate limit" in exc_str


def _is_auth_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "auth" in name or "permissiondenied" in name:
        return True
    auth_tokens = (
        "authentication",
        "unauthorized",
        "permission denied",
        "invalid api key",
        "api key",
        "http 401",
        "status code 401",
    )
    return any(token in msg for token in auth_tokens)


def _load_config(config: dict | str | Path | None) -> dict:
    """Resolve config input to a merged config dict."""
    from miniprophet.config import get_config_from_spec
    from miniprophet.utils.serialize import recursive_merge

    defaults = get_config_from_spec("default")

    if config is None:
        return defaults
    if isinstance(config, str | Path):
        user = get_config_from_spec(config)
    elif isinstance(config, dict):
        user = config
    else:
        raise TypeError(f"config must be dict, str, Path, or None — got {type(config).__name__}")

    return recursive_merge(defaults, user)


async def _process_problem(
    problem: ForecastProblem,
    *,
    config: dict,
    model: Any,
    search_backend: Any,
    coordinator: RateLimitCoordinator,
    progress: BatchProgressCallback | None,
    timeout_seconds: float,
    max_total_cost: float,
    total_cost_ref: list[float],
    cost_lock: asyncio.Lock,
    agent_name: str | None,
    agent_class: type | None,
    agent_kwargs: dict[str, Any],
    search_date_before: str | None,
    search_date_after: str | None,
    include_trajectory: bool,
) -> tuple[ForecastResult, bool]:
    """Process one problem. Returns (result, done). done=False means retry."""
    from miniprophet.agent.context import get_context_manager
    from miniprophet.environment.forecast_env import ForecastEnvironment, create_default_tools
    from miniprophet.environment.source_registry import SourceRegistry

    task_id = problem.task_id
    result = ForecastResult(task_id=task_id, title=problem.title)

    if progress is not None:
        progress.on_run_start(task_id)

    agent: Any = None

    try:
        if max_total_cost > 0 and total_cost_ref[0] >= max_total_cost:
            result.status = "skipped_cost_limit"
            result.error = f"Total cost limit (${max_total_cost:.2f}) reached."
            if progress is not None:
                progress.on_run_end(task_id, result.status)
            return result, True

        search_cfg = config.get("search", {})
        agent_cfg = config.get("agent", {})
        agent_search_limit = int(agent_cfg.get("search_limit", 10) or 10)
        max_gist = int(search_cfg.get("max_source_display_chars", 200) or 200)
        registry = SourceRegistry(max_gist_chars=max_gist)
        tools = create_default_tools(
            search_tool=search_backend,
            registry=registry,
            search_limit=agent_search_limit,
            search_results_limit=search_cfg.get("search_results_limit", 5),
        )
        env = ForecastEnvironment(tools, registry=registry)

        cm_cfg = config.get("context_manager", {})
        ctx_mgr = get_context_manager(cm_cfg)

        resolved_kwargs = dict(agent_kwargs)
        if agent_class is None:
            resolved_kwargs = {**agent_cfg, **resolved_kwargs}

        agent = EvalAgentFactory.create(
            model=model,
            env=env,
            context_manager=ctx_mgr,
            agent_name=agent_name,
            agent_class=agent_class,
            agent_kwargs=resolved_kwargs,
            task_id=task_id,
            coordinator=coordinator,
            progress_manager=progress,
        )

        runtime_kwargs: dict[str, Any] = {
            "search_date_before": search_date_before or problem.predict_by,
            "search_date_after": search_date_after,
        }

        forecast = await asyncio.wait_for(
            agent.run(
                title=problem.title,
                ground_truth=problem.ground_truth,
                **runtime_kwargs,
            ),
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        )

        result.status = forecast.get("exit_status", "unknown")
        result.submission = forecast.get("submission") or None
        result.evaluation = forecast.get("evaluation") or None
        result.cost = {
            "model": float(getattr(agent, "model_cost", 0.0) or 0.0),
            "search": float(getattr(agent, "search_cost", 0.0) or 0.0),
            "total": float(getattr(agent, "total_cost", 0.0) or 0.0),
        }

        async with cost_lock:
            total_cost_ref[0] += result.cost["total"]

        if include_trajectory:
            save_fn = getattr(getattr(agent, "_agent", agent), "save", None)
            if callable(save_fn):
                try:
                    import tempfile

                    with tempfile.TemporaryDirectory() as tmpdir:
                        traj = save_fn(Path(tmpdir))
                        if isinstance(traj, dict):
                            result.trajectory = traj
                except Exception:
                    logger.debug("Failed to capture trajectory for %s", task_id, exc_info=True)

        if progress is not None:
            progress.on_run_end(task_id, result.status)
        return result, True

    except SearchAuthError as exc:
        result.status = "auth_error"
        result.error = str(exc)
        if progress is not None:
            progress.on_run_end(task_id, result.status)
        raise BatchFatalError(f"Run {task_id} failed with auth error: {exc}") from exc

    except TimeoutError:
        result.status = "BatchRunTimeoutError"
        result.error = f"Run timed out after {timeout_seconds:.1f}s"
        if progress is not None:
            progress.on_run_end(task_id, result.status)
        return result, True

    except Exception as exc:
        if _is_rate_limit_error(exc):
            await coordinator.signal_rate_limit()
            if problem.retries < MAX_RETRIES:
                problem.retries += 1
                logger.warning(
                    "Rate limit for %s (attempt %d/%d) -- will retry.",
                    task_id,
                    problem.retries,
                    MAX_RETRIES,
                )
                if progress is not None:
                    progress.on_run_end(task_id, f"rate_limited (retry {problem.retries})")
                return result, False

        if _is_auth_error(exc):
            result.status = "auth_error"
            result.error = str(exc)
            if progress is not None:
                progress.on_run_end(task_id, result.status)
            raise BatchFatalError(f"Run {task_id} failed with auth error: {exc}") from exc

        result.status = type(exc).__name__
        result.error = str(exc)
        logger.error("Run %s failed: %s", task_id, exc, exc_info=True)
        if progress is not None:
            progress.on_run_end(task_id, result.status)
        return result, True


async def batch_forecast(
    problems: list[ForecastProblem],
    config: dict | str | Path | None = None,
    *,
    workers: int = 1,
    timeout_seconds: float = 180.0,
    max_total_cost: float = 0.0,
    search_date_before: str | None = None,
    search_date_after: str | None = None,
    on_progress: BatchProgressCallback | None = None,
    include_trajectory: bool = False,
    agent_name: str | None = None,
    agent_class: type | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    model: Any | None = None,
    search_backend: Any | None = None,
) -> list[ForecastResult]:
    """Run forecasting problems concurrently and return results.

    Async-first implementation using asyncio.Semaphore + gather.
    """
    resolved_config = _load_config(config)
    resolved_kwargs = dict(agent_kwargs or {})

    if model is None:
        from miniprophet.models import get_model

        model = get_model(config=resolved_config.get("model", {}))
    if search_backend is None:
        from miniprophet.tools.search import get_search_backend

        search_backend = get_search_backend(search_cfg=resolved_config.get("search", {}))

    coordinator = RateLimitCoordinator()
    cost_lock = asyncio.Lock()
    total_cost_ref = [0.0]

    work_problems = copy.deepcopy(problems)
    results_map: dict[str, ForecastResult] = {}
    semaphore = asyncio.Semaphore(workers)

    async def run_one(problem: ForecastProblem) -> None:
        async with semaphore:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    await coordinator.wait_if_paused()
                    result, done = await _process_problem(
                        problem,
                        config=resolved_config,
                        model=model,
                        search_backend=search_backend,
                        coordinator=coordinator,
                        progress=on_progress,
                        timeout_seconds=timeout_seconds,
                        max_total_cost=max_total_cost,
                        total_cost_ref=total_cost_ref,
                        cost_lock=cost_lock,
                        agent_name=agent_name,
                        agent_class=agent_class,
                        agent_kwargs=resolved_kwargs,
                        search_date_before=search_date_before,
                        search_date_after=search_date_after,
                        include_trajectory=include_trajectory,
                    )
                    results_map[problem.task_id] = result
                    if done:
                        return
                except BatchFatalError:
                    raise
                except Exception as exc:
                    fr = ForecastResult(
                        task_id=problem.task_id,
                        title=problem.title,
                        status=type(exc).__name__,
                        error=str(exc),
                    )
                    results_map[problem.task_id] = fr
                    if on_progress is not None:
                        on_progress.on_run_end(problem.task_id, fr.status)
                    logger.error(
                        "Unhandled worker exception for %s",
                        problem.task_id,
                        exc_info=True,
                    )
                    return

    tasks = [asyncio.create_task(run_one(p)) for p in work_problems]

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Check for BatchFatalError
        for r in results:
            if isinstance(r, BatchFatalError):
                raise r
    except KeyboardInterrupt:
        for t in tasks:
            t.cancel()
        logger.info("batch_forecast interrupted by user.")

    return [
        results_map.get(p.task_id, ForecastResult(task_id=p.task_id, title=p.title))
        for p in problems
    ]


def batch_forecast_sync(
    problems: list[ForecastProblem],
    config: dict | str | Path | None = None,
    *,
    workers: int = 1,
    timeout_seconds: float = 180.0,
    max_total_cost: float = 0.0,
    search_date_before: str | None = None,
    search_date_after: str | None = None,
    on_progress: BatchProgressCallback | None = None,
    include_trajectory: bool = False,
    agent_name: str | None = None,
    agent_class: type | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    model: Any | None = None,
    search_backend: Any | None = None,
) -> list[ForecastResult]:
    """Run forecasting problems in parallel and return results.

    Sync wrapper around :func:`batch_forecast`.

    Args:
        problems: List of forecasting problems to solve.
        config: Config dict, path to yaml, or None for defaults.
        workers: Number of parallel workers.
        timeout_seconds: Per-problem timeout.
        max_total_cost: Stop when cumulative cost exceeds this (0 = unlimited).
        search_date_before: Restrict searches to before this date.
        search_date_after: Restrict searches to after this date.
        on_progress: Optional callback for progress updates.
        include_trajectory: If True, include agent trajectory in results.
        agent_name: Built-in agent name (default: "default").
        agent_class: Agent class directly.
        agent_kwargs: Extra kwargs forwarded to the agent constructor.
        model: Pre-constructed model instance satisfying the Model protocol.
            When provided, ``get_model()`` is skipped and this model is shared
            across all workers.  Useful for injecting an external LLM adapter.
        search_backend: Pre-constructed search backend instance.
            When provided, ``get_search_backend()`` is skipped and this backend
            is shared across all workers.

    Returns:
        List of ForecastResult, one per input problem, in the same order.
    """
    return asyncio.run(
        batch_forecast(
            problems,
            config,
            workers=workers,
            timeout_seconds=timeout_seconds,
            max_total_cost=max_total_cost,
            search_date_before=search_date_before,
            search_date_after=search_date_after,
            on_progress=on_progress,
            include_trajectory=include_trajectory,
            agent_name=agent_name,
            agent_class=agent_class,
            agent_kwargs=agent_kwargs,
            model=model,
            search_backend=search_backend,
        )
    )
