"""Public batch forecasting API."""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from miniprophet.eval.agent_factory import EvalAgentFactory
from miniprophet.eval.agent_runtime import RateLimitCoordinator
from miniprophet.eval.types import BatchProgressCallback, ForecastProblem, ForecastResult
from miniprophet.exceptions import (
    BatchFatalError,
    BatchRunTimeoutError,
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


def _run_agent_with_timeout(
    *,
    agent: Any,
    timeout_seconds: float,
    cancel_event: threading.Event,
    title: str,
    outcomes: list[str],
    ground_truth: dict[str, int] | None,
    runtime_kwargs: dict[str, Any],
) -> Any:
    timer = None
    if timeout_seconds > 0:
        timer = threading.Timer(timeout_seconds, cancel_event.set)
        timer.daemon = True
        timer.start()
    try:
        result = agent.run(
            title=title,
            outcomes=outcomes,
            ground_truth=ground_truth,
            **runtime_kwargs,
        )
    except BatchRunTimeoutError:
        raise
    finally:
        if timer is not None:
            timer.cancel()
    if cancel_event.is_set():
        raise BatchRunTimeoutError(f"Run timed out after {timeout_seconds:.1f}s")
    return result


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


def _process_problem(
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
    cost_lock: threading.Lock,
    agent_name: str | None,
    agent_import_path: str | None,
    agent_class: type | None,
    agent_kwargs: dict[str, Any],
    search_date_before: str | None,
    search_date_after: str | None,
    include_trajectory: bool,
) -> tuple[ForecastResult, bool]:
    """Process one problem. Returns (result, done). done=False means retry."""
    from miniprophet.agent.context import get_context_manager
    from miniprophet.environment.forecast_env import ForecastEnvironment, create_default_tools
    from miniprophet.environment.source_board import SourceBoard

    task_id = problem.task_id
    result = ForecastResult(task_id=task_id, title=problem.title)

    if progress is not None:
        progress.on_run_start(task_id)

    agent: Any = None
    cancel_event = threading.Event()

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
        board = SourceBoard()
        tools = create_default_tools(
            search_tool=search_backend,
            outcomes=problem.outcomes,
            board=board,
            search_limit=agent_search_limit,
            search_results_limit=search_cfg.get("search_results_limit", 5),
            max_source_display_chars=search_cfg.get("max_source_display_chars", 2000),
        )
        env = ForecastEnvironment(tools, board=board)

        cm_cfg = config.get("context_manager", {})
        ctx_mgr = get_context_manager(cm_cfg)

        resolved_kwargs = dict(agent_kwargs)
        if not agent_import_path and agent_class is None:
            resolved_kwargs = {**agent_cfg, **resolved_kwargs}

        agent = EvalAgentFactory.create(
            model=model,
            env=env,
            context_manager=ctx_mgr,
            agent_name=agent_name,
            agent_import_path=agent_import_path,
            agent_class=agent_class,
            agent_kwargs=resolved_kwargs,
            task_id=task_id,
            coordinator=coordinator,
            progress_manager=progress,
            cancel_event=cancel_event,
        )

        runtime_kwargs: dict[str, Any] = {
            "search_date_before": search_date_before or problem.predict_by,
            "search_date_after": search_date_after,
        }

        forecast = _run_agent_with_timeout(
            agent=agent,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
            title=problem.title,
            outcomes=problem.outcomes,
            ground_truth=problem.ground_truth,
            runtime_kwargs=runtime_kwargs,
        )

        result.status = forecast.get("exit_status", "unknown")
        result.submission = forecast.get("submission") or None
        result.evaluation = forecast.get("evaluation") or None
        result.cost = {
            "model": float(getattr(agent, "model_cost", 0.0) or 0.0),
            "search": float(getattr(agent, "search_cost", 0.0) or 0.0),
            "total": float(getattr(agent, "total_cost", 0.0) or 0.0),
        }

        with cost_lock:
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

    except BatchRunTimeoutError as exc:
        result.status = type(exc).__name__
        result.error = str(exc)
        if progress is not None:
            progress.on_run_end(task_id, result.status)
        return result, True

    except Exception as exc:
        if _is_rate_limit_error(exc):
            coordinator.signal_rate_limit()
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


def batch_forecast(
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
    agent_import_path: str | None = None,
    agent_class: type | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    model: Any | None = None,
) -> list[ForecastResult]:
    """Run forecasting problems in parallel and return results.

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
        agent_import_path: Import path for custom agent ("module.path:ClassName").
        agent_class: Agent class directly (alternative to agent_import_path).
        agent_kwargs: Extra kwargs forwarded to the agent constructor.
        model: Pre-constructed model instance satisfying the Model protocol.
            When provided, ``get_model()`` is skipped and this model is shared
            across all workers.  Useful for injecting an external LLM adapter.

    Returns:
        List of ForecastResult, one per input problem, in the same order.
    """
    from miniprophet.tools.search import get_search_backend

    resolved_config = _load_config(config)
    resolved_kwargs = dict(agent_kwargs or {})

    if model is None:
        from miniprophet.models import get_model

        model = get_model(config=resolved_config.get("model", {}))
    search_backend = get_search_backend(search_cfg=resolved_config.get("search", {}))

    coordinator = RateLimitCoordinator()
    cost_lock = threading.Lock()
    total_cost_ref = [0.0]
    fatal_event = threading.Event()
    fatal_error: list[str | None] = [None]

    # Deep-copy problems so retries mutation is local
    work_problems = copy.deepcopy(problems)

    results_map: dict[str, ForecastResult] = {}
    results_lock = threading.Lock()

    queue: Queue[ForecastProblem] = Queue()
    for p in work_problems:
        queue.put(p)

    def worker_loop() -> None:
        while True:
            if fatal_event.is_set():
                return
            try:
                problem = queue.get(timeout=0.5)
            except Empty:
                if queue.empty() or fatal_event.is_set():
                    return
                continue
            try:
                if fatal_event.is_set():
                    queue.task_done()
                    return
                result, done = _process_problem(
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
                    agent_import_path=agent_import_path,
                    agent_class=agent_class,
                    agent_kwargs=resolved_kwargs,
                    search_date_before=search_date_before,
                    search_date_after=search_date_after,
                    include_trajectory=include_trajectory,
                )
                with results_lock:
                    results_map[problem.task_id] = result
                if not done and not fatal_event.is_set():
                    queue.put(problem)
            except BatchFatalError as exc:
                fatal_error[0] = str(exc)
                fatal_event.set()
                # Drain queue
                while True:
                    try:
                        queue.get_nowait()
                        queue.task_done()
                    except Empty:
                        break
                queue.task_done()
                return
            except Exception as exc:
                fr = ForecastResult(
                    task_id=problem.task_id,
                    title=problem.title,
                    status=type(exc).__name__,
                    error=str(exc),
                )
                with results_lock:
                    results_map[problem.task_id] = fr
                if on_progress is not None:
                    on_progress.on_run_end(problem.task_id, fr.status)
                logger.error("Unhandled worker exception for %s", problem.task_id, exc_info=True)
            finally:
                queue.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_loop) for _ in range(workers)]
        try:
            queue.join()
        except KeyboardInterrupt:
            logger.info("batch_forecast interrupted by user.")
        for f in futures:
            try:
                f.result(timeout=1)
            except Exception:
                pass

    if fatal_event.is_set():
        raise BatchFatalError(fatal_error[0] or "Batch terminated due to a fatal error.")

    # Return results in the same order as input problems
    return [
        results_map.get(p.task_id, ForecastResult(task_id=p.task_id, title=p.title))
        for p in problems
    ]
